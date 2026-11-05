from curl_cffi import requests as cffi_requests
import requests
import sys
import time
import re

def main(zip_path):
    with open('.r2ai_cred') as f:
        lines = f.read().splitlines()
        user = lines[0].split('=')[1]
        pw = lines[1].split('=')[1]

    s_cffi = cffi_requests.Session(impersonate="chrome110")
    r = s_cffi.get("https://leaderboard.aiguru.com.vn/accounts/login/")
    m = re.search(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"', r.text)
    csrf = m.group(1) if m else s_cffi.cookies.get('csrftoken')
    
    data = {'csrfmiddlewaretoken': csrf, 'username': user, 'password': pw, 'next': '/'}
    r2 = s_cffi.post("https://leaderboard.aiguru.com.vn/accounts/login/", data=data, headers={'Referer': 'https://leaderboard.aiguru.com.vn/accounts/login/'})
    
    session_id = s_cffi.cookies.get('sessionid')
    csrftoken = s_cffi.cookies.get('csrftoken')
    if not session_id:
        print("Login failed, cookies:", s_cffi.cookies.get_dict())
        return
        
    print("Logged in. sessionid:", session_id)
    
    # Switch to standard requests for multipart upload
    s = requests.Session()
    s.cookies.set('sessionid', session_id)
    s.cookies.set('csrftoken', csrftoken)
    s.headers.update({
        'X-CSRFToken': csrftoken,
        'Referer': 'https://leaderboard.aiguru.com.vn/competitions/15/'
    })
    
    print(f"Submitting {zip_path}...")
    with open(zip_path, 'rb') as f:
        files = {'data': (zip_path, f, 'application/zip')}
        data = {'phase': 44}
        r3 = s.post('https://leaderboard.aiguru.com.vn/api/submissions/', data=data, files=files)
        print("Submit response:", r3.status_code, r3.text)
        
        if r3.status_code == 201:
            sub_id = r3.json().get('id')
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

if __name__ == "__main__":
    main(sys.argv[1])
