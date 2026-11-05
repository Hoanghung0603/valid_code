# Long Châu ID lẻ bằng Crawl4AI

Luồng này dùng đúng `id` số lẻ trong bảng `crawl_status` của `status.duckdb`.
ID là khóa của DB, không phải số trong URL. Nguồn DB chỉ được đọc qua snapshot;
output và cooldown dùng chung với luồng Long Châu hiện có.

## 1. Kiểm tra môi trường, không truy cập Long Châu

```powershell
cd 'D:\New folder (3)\r2aiIComeForFood26'
.\.venv\Scripts\python.exe -c "import crawl4ai; print('Crawl4AI import OK')"
.\.venv\Scripts\python.exe .\scripts\check_crawl4ai.py
```

Nếu thiếu trình duyệt Chromium, cài bằng:

```powershell
.\.venv\Scripts\python.exe -m playwright install chromium
```

## 2. Lập kế hoạch offline

```powershell
.\scripts\run_local.ps1 longchau-odd --engine crawl4ai --status-db '.\status.duckdb'
Get-Content 'D:\r2ai_data\crawl\longchau_odd\cooldown.json' -Raw
```

Lệnh đầu chỉ đếm URL, không gửi request. Nếu dùng DB ở nơi khác, thay đường dẫn
`--status-db`. Tôn trọng `next_probe_at` trong `cooldown.json`; không xóa file này.

## 3. Thử đúng một trang

```powershell
.\scripts\run_local.ps1 longchau-odd --engine crawl4ai --status-db '.\status.duckdb' --probe
```

Probe chỉ mở một URL nguồn, nhưng trình duyệt có thể tải thêm `robots.txt` và
tài nguyên trang. Nếu kết quả là `"ok": 1`, có thể quét liên tục toàn bộ hàng đợi:

```powershell
.\scripts\run_local.ps1 longchau-odd --engine crawl4ai --status-db '.\status.duckdb' --scan --continuous --batch-size 50 --delay 30
```

Nếu 2.717 ID lẻ thấp nhất đã có ở bộ dữ liệu khác, thêm
`--skip-first-odd 2717` vào lệnh lập kế hoạch và lệnh quét. Với DB hiện tại,
ID đầu tiên sau phần bỏ qua là `209709`. Các ID đã lưu trong output của lượt
quét này vẫn được kiểm tra và bỏ qua khi chạy lại.

Mỗi URL nguồn chạy tuần tự; không retry, proxy, stealth hay giải CAPTCHA.
Crawl4AI bật kiểm tra `robots.txt`. Gặp 401/403/429/503 hoặc trang xác minh,
luồng dừng và giữ cooldown ít nhất 6 giờ. `--continuous` lưu từng 50 URL nhưng
không nghỉ cố định 3 giờ khi vẫn nhận trang hợp lệ. Với gần 40.000 URL và
30 giây mỗi URL, lượt chạy có thể kéo dài khoảng 14 ngày nếu không gián đoạn.
Muốn quét tiếp, chạy lại cùng lệnh; ID đã lưu sẽ được bỏ qua. Nếu bị chặn,
thử lại `--probe` sau cooldown trước.

## Output và dừng mềm

`D:\r2ai_data\crawl\longchau_odd\html\home_shard_*.jsonl.zst` chứa HTML
do Crawl4AI trả về, mã hóa UTF-8 theo định dạng sáu trường của pipeline.
Đây là HTML snapshot của trang, không phải bytes phản hồi HTTP nguyên thủy.
`status.csv`, `run_summary.json`, `cooldown.json` và `crawl.log` ở cùng thư mục.

Tạo file `D:\r2ai_data\crawl\longchau_odd\STOP` để dừng mềm. Xóa riêng file
`STOP` sau khi job thoát nếu muốn chạy tiếp. Không xóa cooldown hoặc reset DB.
Sau khi có shard thành công, có thể dùng lệnh `merge-home` trong
`LENH_CHAY_LOCAL.md` để nhập vào pipeline.
