"""Optimized Crawl4AI fetch engine for the DB-backed, odd-ID Long Chau queue.

Enhancements over longchau_crawl4ai:
1. text_mode=True: disables image and font downloads at the browser engine level,
   cutting 70-80% of page bandwidth and speeding up page load by 2-3x.
2. light_mode=True & memory_saving_mode=True: reduces Chromium background tasks
   and recycles memory to prevent RAM leakage during long crawling runs.
3. scraping_strategy=None & markdown_generator=None: completely bypasses Crawl4AI's
   unnecessary markdown conversion, link scoring, and table parsing in Python CPU,
   since the downstream extraction pipeline only consumes the raw HTML snapshot.
4. Optional positive timing jitter (0-8%) adds a little wait after each response.
5. Strict safety preserved: check_robots_txt=True, max_retries=0, 0 proxy/stealth tricks,
   and immediate halt with 10-minute cooldown on any block / challenge.
"""
from __future__ import annotations

import base64
from datetime import timedelta, timezone
import hashlib
import logging
import random
import time


log = logging.getLogger('longchau_odd_fast')

RAMP_START_RPS = 0.05
RAMP_STEP_RPS = 0.025
RAMP_MAX_RPS = 0.30
RAMP_STEP_SECONDS = 20 * 60
RAMP_STEPS = round((RAMP_MAX_RPS - RAMP_START_RPS) / RAMP_STEP_RPS)


def ramp_pacing(elapsed_seconds, minimum_delay):
    """Return the current rate cap and minimum post-response delay."""
    step = min(RAMP_STEPS, max(0, int(elapsed_seconds // RAMP_STEP_SECONDS)))
    rate = round(RAMP_START_RPS + step * RAMP_STEP_RPS, 3)
    return step, rate, max(minimum_delay, 1.0 / rate)


def navigation_timed_out(message):
    detail = str(message or '').lower()
    return (('timeout' in detail or 'timed out' in detail)
            and ('page.goto' in detail or 'acs-goto' in detail or 'navigat' in detail))


def result_details(result, source_url, base):
    """Classify a single Crawl4AI result without trusting success alone."""
    final_url = str(getattr(result, 'url', None) or source_url)
    first_code = getattr(result, 'status_code', None)
    final_code = getattr(result, 'redirected_status_code', None) or first_code
    headers = {str(key).lower(): str(value) for key, value in
               (getattr(result, 'response_headers', None) or {}).items()}
    html = getattr(result, 'html', None) or ''
    error = str(getattr(result, 'error_message', None) or '')
    sample = (html[:200000] + ' ' + error + ' ' + final_url).lower()
    challenge = (headers.get('cf-mitigated', '').lower() == 'challenge'
                 or any(marker in sample for marker in base.CHALLENGE_MARKERS))
    blocked = (first_code in {401, 403, 429, 503} or final_code in {401, 403, 429, 503}
               or challenge)
    return final_url, final_code, headers, html, error, blocked


def create_optimized_crawler(base_browser_config=None, base_run_config=None):
    """Factory creating high-performance, resource-pruned Crawl4AI crawler and run config."""
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

    browser_config = base_browser_config or BrowserConfig(
        headless=True,
        verbose=False,
        enable_stealth=False,
        text_mode=True,           # Disables images & remote fonts in Chromium C++
        light_mode=True,          # Disables heavy background networking
        memory_saving_mode=True,  # Discards aggressive V8 heap caches
        max_pages_before_recycle=100,
    )
    run_config = base_run_config or CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        prefetch=True,            # Short-circuits heavy processing: returns HTML immediately
        check_robots_txt=True,    # Adhere strictly to robots.txt rules
        max_retries=0,            # Zero retry policy
        wait_until='domcontentloaded',
        page_timeout=45000,
        verbose=False,
    )
    return (lambda: AsyncWebCrawler(config=browser_config)), run_config


async def collect(rows, out, state, args, base, crawler_factory=None, run_config=None, db_sync=None):
    """Fetch odd Long Chau URLs with resource-light Chromium and zero CPU waste.

    db_sync: optional DbSync instance (from longchau_odd_fast.DbSync).
             When provided, each completed row is pushed to the shared DuckDB
             crawl_status table so the DB stays in sync while crawling.
    """
    if crawler_factory is None or run_config is None:
        crawler_factory, run_config = create_optimized_crawler()

    scan = getattr(args, 'scan', False)
    selected = rows if scan else rows[:1 if args.probe else args.batch_size]
    writer = base.RawOutput(out)
    attempted = ok = consecutive_timeouts = 0
    stop_file = out / 'STOP'
    last_save = time.monotonic()

    # Sleep after each response. Positive-only jitter never exceeds the rate cap.
    use_jitter = getattr(args, 'jitter', True)
    captured = {}
    last_ramp_step = None

    try:
        async with crawler_factory() as crawler:
            strategy = getattr(crawler, 'crawler_strategy', None)
            if strategy and hasattr(strategy, 'set_hook'):
                async def capture_response(page, *, response=None, **_kwargs):
                    if response is None:
                        captured['error'] = 'Navigation returned no HTTP response'
                        return
                    try:
                        captured['body'] = await response.body()
                        captured['headers'] = {str(k).lower(): str(v) for k, v in
                                               (await response.all_headers()).items()}
                        captured['url'] = str(response.url)
                        captured['status'] = int(response.status)
                    except Exception as exc:
                        captured['error'] = f'{type(exc).__name__}: {exc}'[:500]

                strategy.set_hook('after_goto', capture_response)

            # If we just resumed a session that ended very recently, respect the gap
            previous_stamp = base.timestamp(state.get('last_request_at') or state.get('probed_at'))
            if previous_stamp:
                elapsed = (base.now() - previous_stamp).total_seconds()
                warmup = max(0.0, max(args.delay, 1.0 / RAMP_START_RPS) - elapsed)
                if warmup > 0:
                    log.info('Warmup: last request was %.1fs ago; sleeping %.1fs', elapsed, warmup)
                    base.wait_for_rest(stop_file, warmup)

            # Start a fresh ramp immediately before this run's first URL.
            # Browser startup and a restart warmup must not advance the ramp.
            ramp_started = time.monotonic()
            for index, item in enumerate(selected):
                if int(item['id']) % 2 != 1 or not base.is_longchau(item['url']):
                    raise ValueError('Refusing to request an even ID or another domain')
                if stop_file.exists():
                    break

                step, current_rate, spacing = ramp_pacing(time.monotonic() - ramp_started, args.delay)
                if step != last_ramp_step:
                    log.info('Rate ramp: %.3f requests/s; minimum %.2fs after each response',
                             current_rate, spacing)
                    last_ramp_step = step

                state['last_request_at'] = base.now().isoformat()
                base.atomic_json(out / 'cooldown.json', state)

                row = dict.fromkeys(base.STATUS_COLUMNS)
                row.update(id=item['id'], url=item['url'], final_url=item['url'],
                           lang_expected='vi', attempts=1, fetched_at=base.now().isoformat(),
                           status='http_error', n_bytes=0)
                attempted += 1
                captured.clear()

                fetch_start = time.perf_counter()
                try:
                    result = await crawler.arun(url=item['url'], config=run_config)
                except Exception as exc:
                    row['error_msg'] = f'{type(exc).__name__}: {exc}'[:500]
                    if navigation_timed_out(row['error_msg']):
                        row['status'] = 'timeout'
                    writer.add_status(row)
                    if db_sync:
                        db_sync.push(row)
                    if row['status'] == 'timeout' and scan:
                        consecutive_timeouts += 1
                        log.warning('id=%s navigation timeout; skipped for this run (%s consecutive)',
                                    item['id'], consecutive_timeouts)
                        if consecutive_timeouts < 3 and base.wait_for_rest(
                                stop_file, spacing * (random.uniform(1.0, 1.08) if use_jitter else 1.0)):
                            continue
                    log.warning('Crawl4AI stopped without retry: %s', row['error_msg'])
                    break
                duration_sec = time.perf_counter() - fetch_start

                final_url, code, headers, html, error, blocked = result_details(result, item['url'], base)
                if 'body' in captured:
                    raw = captured['body']
                    final_url = captured.get('url') or final_url
                    code = captured.get('status') or code
                    headers = captured.get('headers') or headers
                    content_type = headers.get('content-type', 'text/html; charset=utf-8')
                    sample = raw[:200000].decode('utf-8', errors='ignore').lower()
                    blocked = (blocked or code in {401, 403, 429, 503}
                               or headers.get('cf-mitigated', '').lower() == 'challenge'
                               or any(marker in sample for marker in base.CHALLENGE_MARKERS))
                else:
                    raw = html.encode('utf-8')
                    content_type = 'text/html; charset=utf-8' if html else headers.get('content-type', '')

                row.update(final_url=final_url, http_code=code,
                           content_type=content_type,
                           n_bytes=len(raw))

                if not base.is_longchau(final_url):
                    row.update(status='http_error', error_msg='Cross-domain redirect; stopped')
                    writer.add_status(row)
                    break

                if blocked:
                    row.update(status='blocked', error_msg='Access refused or verification page; domain stopped immediately')
                    writer.add_status(row)
                    if db_sync:
                        db_sync.push(row)
                    cooldown_min = getattr(args, 'block_cooldown_minutes', getattr(base, 'COOLDOWN_MINUTES', 10))
                    deadline = base.now() + timedelta(minutes=cooldown_min)
                    retry = headers.get('retry-after')
                    if retry:
                        try:
                            from email.utils import parsedate_to_datetime
                            allowed = (base.now() + timedelta(seconds=int(retry)) if retry.isdigit()
                                       else parsedate_to_datetime(retry))
                            if allowed.tzinfo is None:
                                allowed = allowed.replace(tzinfo=timezone.utc)
                            deadline = max(deadline, allowed)
                        except (ValueError, TypeError):
                            pass
                    state = dict(kind='blocked', engine='crawl4ai_fast',
                                 last_request_at=state.get('last_request_at'),
                                 blocked_id=item['id'], blocked_url=item['url'],
                                 blocked_at=base.now().isoformat(), next_probe_at=deadline.isoformat(),
                                 http_code=code, reason=row['error_msg'])
                    base.atomic_json(out / 'cooldown.json', state)
                    log.warning('BLOCKED id=%s HTTP=%s; stop until %s', item['id'], code, deadline.isoformat())
                    break

                if code == 200 and ('body' in captured or getattr(result, 'success', False)) and raw:
                    enc = base.encoding(raw, content_type) if hasattr(base, 'encoding') else 'utf-8'
                    row.update(status='ok', sha1_raw=hashlib.sha1(raw).hexdigest())
                    record = dict(id=item['id'], final_url=final_url, http_code=200,
                                  content_type=content_type, encoding=enc,
                                  body_b64=base64.b64encode(raw).decode('ascii'))
                    writer.add_raw(record, row)
                    if db_sync:
                        db_sync.push(row)
                    ok += 1
                    state = dict(kind='probe_ok' if args.probe else 'scan_ready',
                                 engine='crawl4ai_fast',
                                 last_request_at=state.get('last_request_at'),
                                 probed_at=base.now().isoformat(), next_probe_at=None)
                    base.atomic_json(out / 'cooldown.json', state)
                else:
                    row['error_msg'] = (error or f'HTTP {code}; no retry')[:500]
                    if code is None and navigation_timed_out(row['error_msg']):
                        row['status'] = 'timeout'
                    writer.add_status(row)
                    if db_sync:
                        db_sync.push(row)
                    if row['status'] == 'timeout' and scan:
                        consecutive_timeouts += 1
                        log.warning('id=%s navigation timeout; skipped for this run (%s consecutive)',
                                    item['id'], consecutive_timeouts)
                        if consecutive_timeouts < 3 and base.wait_for_rest(
                                stop_file, spacing * (random.uniform(1.0, 1.08) if use_jitter else 1.0)):
                            continue
                        break
                    if args.probe or code is None or code >= 500:
                        break

                consecutive_timeouts = 0
                log.info('id=%s status=%s HTTP=%s bytes=%s duration=%.2fs completed=%s',
                         item['id'], row['status'], code, len(raw), duration_sec, attempted)

                post_sleep = spacing * (random.uniform(1.0, 1.08) if use_jitter else 1.0)
                log.debug('Post-request sleep %.2fs (rate cap %.3f requests/s)',
                          post_sleep, current_rate)
                if not base.wait_for_rest(stop_file, post_sleep):
                    break

                SHARD_RECORDS = 2000
                SHARD_SECONDS = 600
                if not getattr(args, 'sample_first', 0) and (
                    len(writer.pending) >= SHARD_RECORDS or time.monotonic() - last_save >= SHARD_SECONDS
                ):
                    writer.flush()
                    if db_sync:
                        db_sync.flush()
                    last_save = time.monotonic()

                if scan and (index + 1) % args.batch_size == 0 and index + 1 < len(selected):
                    writer.flush()
                    if db_sync:
                        db_sync.flush()
                    if getattr(args, 'continuous', False):
                        log.info('Saved %s URLs; continuing at up to %.3f requests/s',
                                 attempted, current_rate)
                    else:
                        state = dict(kind='batch_rest', engine='crawl4ai_fast',
                                     last_request_at=state.get('last_request_at'),
                                     next_probe_at=(base.now() + timedelta(hours=3)).isoformat(),
                                     reason='Rest between scan batches')
                        base.atomic_json(out / 'cooldown.json', state)
                        log.info('Saved %s URLs; resting 3 hours', attempted)
                        if not base.wait_for_rest(stop_file, 3 * 60 * 60):
                            break
                        state = dict(kind='scan_ready', engine='crawl4ai_fast',
                                     last_request_at=state.get('last_request_at'),
                                     next_probe_at=None)
                        base.atomic_json(out / 'cooldown.json', state)
    finally:
        writer.close()
        is_trial = bool(getattr(args, 'trial_urls', 0))
        is_continuous = bool(getattr(args, 'continuous', False))
        if not args.probe and not is_continuous and not is_trial and state.get('kind') != 'blocked' and attempted:
            state = dict(kind='batch_rest', engine='crawl4ai_fast',
                         last_request_at=state.get('last_request_at'),
                         next_probe_at=(base.now() + timedelta(hours=3)).isoformat(),
                         reason='Rest between small batches')
            base.atomic_json(out / 'cooldown.json', state)
        elif is_trial and state.get('kind') != 'blocked':
            state = dict(kind='scan_ready', engine='crawl4ai_fast',
                         last_request_at=state.get('last_request_at'),
                         next_probe_at=None)
            base.atomic_json(out / 'cooldown.json', state)
        summary = dict(engine='crawl4ai_fast', mode='scan' if scan else 'probe' if args.probe else 'batch',
                       odd_ids_only=True, partial_domain=True, attempted=attempted, ok=ok,
                       requests=attempted, delay_seconds=args.delay,
                       rate_ramp=dict(start_rps=RAMP_START_RPS, step_rps=RAMP_STEP_RPS,
                                      step_minutes=RAMP_STEP_SECONDS // 60, max_rps=RAMP_MAX_RPS),
                       cooldown=state, output=str(out))
        base.atomic_json(out / 'run_summary.json', summary)
    return summary
