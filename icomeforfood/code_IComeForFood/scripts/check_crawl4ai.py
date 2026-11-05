"""Offline browser smoke test. Does not request any website."""
import asyncio

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig


async def main():
    browser_config = BrowserConfig(
        headless=True,verbose=False,enable_stealth=False,
    )
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,check_robots_txt=True,max_retries=0,
        wait_until='domcontentloaded',page_timeout=30000,verbose=False,
    )
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url='raw:<html><body><h1>Kiểm tra Crawl4AI</h1></body></html>',
            config=run_config,
        )
    if not result.success or 'Kiểm tra Crawl4AI' not in (result.html or ''):
        raise RuntimeError(f'Crawl4AI browser smoke test failed: {result.error_message}')
    print('Crawl4AI browser OK; local raw HTML processed; no website requested.')


if __name__=='__main__':
    asyncio.run(main())
