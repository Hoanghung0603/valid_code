"""One-page Crawl4AI diagnosis; leaves crawl status and cooldown untouched."""
import asyncio
import json

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig


URL = ('https://nhathuoclongchau.com.vn/bai-viet/'
       '1-bap-ngo-bao-nhieu-calo-an-bap-co-loi-ich-gi-voi-suc-khoe-1.html')


async def main():
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,check_robots_txt=True,max_retries=0,
        wait_until='domcontentloaded',page_timeout=30000,verbose=False,
    )
    async with AsyncWebCrawler(config=BrowserConfig(headless=True,verbose=False)) as crawler:
        result = await crawler.arun(url=URL,config=config)
    html = result.html or ''
    headers = {str(k).lower(): str(v) for k,v in (result.response_headers or {}).items()}
    print(json.dumps({
        'url': result.url,
        'status_code': result.status_code,
        'redirected_status_code': result.redirected_status_code,
        'success': result.success,
        'html_characters': len(html),
        'has_expected_title': '1 bắp ngô bao nhiêu calo' in html.lower(),
        'challenge_header': headers.get('cf-mitigated'),
        'error': (result.error_message or '')[:300],
    },ensure_ascii=False,indent=2))


if __name__=='__main__':
    asyncio.run(main())
