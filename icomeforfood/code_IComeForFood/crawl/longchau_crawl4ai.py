"""Crawl4AI fetch engine for the DB-backed, odd-ID Long Chau queue.

Use through ``longchau_odd.py --engine crawl4ai`` so the existing queue,
cooldown, output format, and domain lock remain shared with the HTTP engine.
"""
from __future__ import annotations

import base64
from datetime import timedelta, timezone
import hashlib
import logging
import time


log = logging.getLogger('longchau_odd')


def result_details(result, source_url, base):
    """Classify a single Crawl4AI result without trusting success alone."""
    final_url = str(getattr(result, 'url', None) or source_url)
    first_code = getattr(result, 'status_code', None)
    final_code = getattr(result, 'redirected_status_code', None) or first_code
    headers = {str(key).lower(): str(value) for key,value in
               (getattr(result, 'response_headers', None) or {}).items()}
    html = getattr(result, 'html', None) or ''
    error = str(getattr(result, 'error_message', None) or '')
    sample = (html[:200000]+' '+error+' '+final_url).lower()
    challenge = (headers.get('cf-mitigated','').lower() == 'challenge'
                 or any(marker in sample for marker in base.CHALLENGE_MARKERS))
    blocked = (first_code in {401,403,429,503} or final_code in {401,403,429,503}
               or challenge)
    return final_url,final_code,headers,html,error,blocked


async def collect(rows,out,state,args,base,crawler_factory=None,run_config=None):
    """Fetch one URL at a time; never retry, rotate identity, or ignore blocks."""
    if crawler_factory is None:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
        browser_config = BrowserConfig(
            headless=True, verbose=False, enable_stealth=False,
        )
        crawler_factory = lambda: AsyncWebCrawler(config=browser_config)
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS, check_robots_txt=True, max_retries=0,
            wait_until='domcontentloaded', page_timeout=30000, verbose=False,
        )
    scan = getattr(args,'scan',False)
    selected = rows if scan else rows[:1 if args.probe else args.batch_size]
    writer = base.RawOutput(out)
    attempted = ok = 0
    previous_stamp = base.timestamp(state.get('last_request_at') or state.get('probed_at'))
    previous_request = (time.monotonic()-max(0.0,(base.now()-previous_stamp).total_seconds())
                        if previous_stamp else None)
    stop_file = out/'STOP'
    last_save = time.monotonic()
    try:
        async with crawler_factory() as crawler:
            for index,item in enumerate(selected):
                if int(item['id'])%2!=1 or not base.is_longchau(item['url']):
                    raise ValueError('Refusing to request an even ID or another domain')
                if stop_file.exists():
                    break
                if previous_request is not None:
                    remaining = args.delay-(time.monotonic()-previous_request)
                    if remaining>0 and not base.wait_for_rest(stop_file,remaining):
                        break
                if stop_file.exists():
                    break
                previous_request = time.monotonic()
                state['last_request_at'] = base.now().isoformat()
                base.atomic_json(out/'cooldown.json',state)
                row = dict.fromkeys(base.STATUS_COLUMNS)
                row.update(id=item['id'],url=item['url'],final_url=item['url'],
                           lang_expected='vi',attempts=1,fetched_at=base.now().isoformat(),
                           status='http_error',n_bytes=0)
                attempted += 1
                try:
                    result = await crawler.arun(url=item['url'],config=run_config)
                except Exception as exc:
                    row['error_msg'] = f'{type(exc).__name__}: {exc}'[:500]
                    writer.add_status(row)
                    log.warning('Crawl4AI stopped without retry: %s',row['error_msg'])
                    break
                final_url,code,headers,html,error,blocked = result_details(result,item['url'],base)
                raw = html.encode('utf-8')
                row.update(final_url=final_url,http_code=code,
                           content_type='text/html; charset=utf-8' if html else headers.get('content-type',''),
                           n_bytes=len(raw))
                if not base.is_longchau(final_url):
                    row.update(status='http_error',error_msg='Cross-domain redirect; stopped')
                    writer.add_status(row)
                    break
                if blocked:
                    row.update(status='blocked',error_msg='Access refused or verification page; domain stopped immediately')
                    writer.add_status(row)
                    deadline = base.now()+timedelta(hours=base.COOLDOWN_HOURS)
                    retry = headers.get('retry-after')
                    if retry:
                        try:
                            from email.utils import parsedate_to_datetime
                            allowed = (base.now()+timedelta(seconds=int(retry)) if retry.isdigit()
                                       else parsedate_to_datetime(retry))
                            if allowed.tzinfo is None:
                                allowed = allowed.replace(tzinfo=timezone.utc)
                            deadline = max(deadline,allowed)
                        except (ValueError,TypeError):
                            pass
                    state = dict(kind='blocked',engine='crawl4ai',
                                 last_request_at=state.get('last_request_at'),
                                 blocked_id=item['id'],blocked_url=item['url'],
                                 blocked_at=base.now().isoformat(),next_probe_at=deadline.isoformat(),
                                 http_code=code,reason=row['error_msg'])
                    base.atomic_json(out/'cooldown.json',state)
                    log.warning('BLOCKED id=%s HTTP=%s; stop until %s',item['id'],code,deadline.isoformat())
                    break
                if code==200 and getattr(result,'success',False) and raw:
                    row.update(status='ok',sha1_raw=hashlib.sha1(raw).hexdigest())
                    record = dict(id=item['id'],final_url=final_url,http_code=200,
                                  content_type='text/html; charset=utf-8',encoding='utf-8',
                                  body_b64=base64.b64encode(raw).decode('ascii'))
                    writer.add_raw(record,row)
                    ok += 1
                    state = dict(kind='probe_ok' if args.probe else 'scan_ready',
                                 engine='crawl4ai',
                                 last_request_at=state.get('last_request_at'),
                                 probed_at=base.now().isoformat(),next_probe_at=None)
                    base.atomic_json(out/'cooldown.json',state)
                else:
                    row['error_msg'] = (error or f'HTTP {code}; no retry')[:500]
                    writer.add_status(row)
                    if args.probe or code is None or code>=500:
                        break
                log.info('id=%s status=%s HTTP=%s bytes=%s completed=%s',
                         item['id'],row['status'],code,len(raw),attempted)
                if time.monotonic()-last_save>=60:
                    writer.flush()
                    last_save = time.monotonic()
                if scan and (index+1)%args.batch_size==0 and index+1<len(selected):
                    writer.flush()
                    if getattr(args,'continuous',False):
                        log.info('Saved %s URLs; continuing at %.1f seconds per URL',attempted,args.delay)
                    else:
                        state = dict(kind='batch_rest',engine='crawl4ai',
                                     last_request_at=state.get('last_request_at'),
                                     next_probe_at=(base.now()+timedelta(hours=3)).isoformat(),
                                     reason='Rest between scan batches')
                        base.atomic_json(out/'cooldown.json',state)
                        log.info('Saved %s URLs; resting 3 hours',attempted)
                        if not base.wait_for_rest(stop_file,3*60*60):
                            break
                        state = dict(kind='scan_ready',engine='crawl4ai',
                                     last_request_at=state.get('last_request_at'),
                                     next_probe_at=None)
                        base.atomic_json(out/'cooldown.json',state)
    finally:
        writer.close()
        if not args.probe and not getattr(args,'continuous',False) and state.get('kind')!='blocked' and attempted:
            state = dict(kind='batch_rest',engine='crawl4ai',
                         last_request_at=state.get('last_request_at'),
                         next_probe_at=(base.now()+timedelta(hours=3)).isoformat(),
                         reason='Rest between small batches')
            base.atomic_json(out/'cooldown.json',state)
        summary = dict(engine='crawl4ai',mode='scan' if scan else 'probe' if args.probe else 'batch',
                       odd_ids_only=True,partial_domain=True,attempted=attempted,ok=ok,
                       requests=attempted,delay_seconds=args.delay,cooldown=state,output=str(out))
        base.atomic_json(out/'run_summary.json',summary)
    return summary
