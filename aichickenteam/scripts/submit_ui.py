import asyncio
from playwright.async_api import async_playwright
import sys
import time

async def main(zip_path):
    async with async_playwright() as p:
        # headless=False to bypass CF
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        with open('.r2ai_cred') as f:
            lines = f.read().splitlines()
            user = lines[0].split('=')[1]
            pw = lines[1].split('=')[1]
            
        print("Logging in...")
        await page.goto("https://leaderboard.aiguru.com.vn/accounts/login/")
        await page.fill('input[name="username"]', user)
        await page.fill('input[name="password"]', pw)
        await page.click('button[type="submit"]')
        
        try:
            await page.wait_for_selector('text="R2AI2026"', timeout=10000)
        except:
            print("wait timeout for login redirect")
            
        print("Logged in. Navigating to competition...")
        await page.goto("https://leaderboard.aiguru.com.vn/competitions/15/")
        
        # Click My Submissions tab
        print("Clicking My Submissions...")
        try:
            await page.click('text="My Submissions"')
        except Exception as e:
            print("Failed to click My Submissions:", e)
            await browser.close()
            return
            
        time.sleep(2)
        
        print(f"Uploading {zip_path}...")
        try:
            # Usually there's an input type=file, or a label
            await page.set_input_files('input[type="file"]', zip_path)
            time.sleep(5) # wait for upload
        except Exception as e:
            print("Upload failed:", e)
            await browser.close()
            return
            
        print("Submitted via UI. Closing.")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
