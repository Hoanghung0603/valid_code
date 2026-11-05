import asyncio
from playwright.async_api import async_playwright
import sys
import time
import requests

async def main(zip_path):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        with open('.r2ai_cred') as f:
            lines = f.read().splitlines()
            user = lines[0].split('=')[1]
            pw = lines[1].split('=')[1]
            
        print("Logging in...")
        await page.goto("https://leaderboard.aiguru.com.vn/accounts/login/")
        await page.fill('input[name="login"]', user)
        await page.fill('input[name="password"]', pw)
        await page.click('button[type="submit"]')
        
        try:
            await page.wait_for_load_state('networkidle', timeout=10000)
        except Exception as e:
            print("wait timeout", e)
        
        cookies = await page.context.cookies()
        session_id = next((c['value'] for c in cookies if c['name'] == 'sessionid'), None)
        csrftoken = next((c['value'] for c in cookies if c['name'] == 'csrftoken'), None)
        
        if not session_id:
            print("Login failed, cookies:", [c['name'] for c in cookies])
            await browser.close()
            return
            
        print("Submitting via API...")
        s = requests.Session()
        s.cookies.set('sessionid', session_id)
        s.cookies.set('csrftoken', csrftoken)
        s.headers.update({
            'X-CSRFToken': csrftoken,
            'Referer': 'https://leaderboard.aiguru.com.vn/competitions/15/'
        })
        
        with open(zip_path, 'rb') as f:
            files = {'data': (zip_path, f, 'application/zip')}
            data = {'phase': 44}
            r = s.post('https://leaderboard.aiguru.com.vn/api/submissions/', data=data, files=files)
            print("Submit response:", r.status_code, r.text)
            if r.status_code == 201:
                sub_id = r.json().get('id')
                print(f"Submitted successfully! ID: {sub_id}")
                for i in range(20):
                    time.sleep(10)
                    r_stat = s.get(f'https://leaderboard.aiguru.com.vn/api/submissions/{sub_id}/get_details/')
                    if r_stat.status_code == 200:
                        status = r_stat.json().get('status')
                        print(f"Status: {status}")
                        if status in ['Finished', 'Failed']:
                            print("Result:", r_stat.text)
                            break
                    else:
                        print("Error checking status:", r_stat.text)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
