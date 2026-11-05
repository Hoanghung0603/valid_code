# Crawl VOV local — cùng quy trình checkpoint với Long Châu

Crawler `crawl/vov.py` lấy ID/URL từ DuckDB, dùng Chrome headless và lưu raw HTML
riêng. Không cần corpus Parquet. Mặc định chỉ `vov.vn` và `www.vov.vn`, lấy cả ID
chẵn/lẻ. VOV2 được để ngoài vì DB hiện ghi đủ 6.899 URL `ok`.

Bản crawl đã hoàn tất và được bàn giao trong
[snapshot VOV 30/10](../crawl/vov_snapshot_2026-10-30/README.md): đủ 9.784
HTML gốc, manifest/status/checksum và lệnh xuất HTML để nhóm trưởng đọc.

Kiểm tra 28/10/2026: HTTP thường nhận 403; Chrome headless lấy được HTML.
Giống luồng Long Châu, crawler lưu toàn bộ bytes HTML phản hồi HTTP 200,
không lọc theo mẫu bài, không bắt buộc `h1`, `.text-long`, số ký tự hay loại media.
Tiêu đề, văn bản, bảng, metadata, thẻ/link ảnh nội dung và infographic đều giữ
nguyên trong HTML. Crawler không làm sạch hoặc trích văn bản ở bước thu thập.

Theo lựa chọn của người dùng, giữ HTML đầy đủ và link ảnh gốc; không lưu riêng
file ảnh (bao gồm ảnh nền) hoặc video, không chép lời thoại hay OCR.
Các thẻ video và tham chiếu ảnh nền vẫn nằm trong HTML gốc để không làm mất dữ liệu.
Trình duyệt có thể tải tài nguyên khi mở trang, nhưng output chỉ ghi HTML/status.

Chỉ dừng với lỗi truy cập, trang xác minh thật hoặc phản hồi HTML rỗng.
Chữ `captcha` trong script/CSS hay trong nội dung bài không tự tạo kết luận chặn.
HTTP 200 có HTML rỗng sẽ lưu bằng chứng trong `inspection/<id>.html/.json`.
`ok` nghĩa là đã thu được HTML, không khẳng định chất lượng trích xuất nội dung.

## Chạy PowerShell

```powershell
cd 'D:\New folder (3)\r2aiIComeForFood26'

# Lập kế hoạch offline, chưa mở trình duyệt hoặc gửi request:
.\scripts\crawl_vov.ps1 --status-db 'D:\r2ai_data\crawl\status.duckdb'

# Thử một URL, đồng thời lưu raw và status nếu thành công:
.\scripts\crawl_vov.ps1 --status-db 'D:\r2ai_data\crawl\status.duckdb' --probe

# Crawl tiếp toàn bộ URL còn thiếu, lưu từng 50 trang hoặc mỗi 60 giây:
.\scripts\crawl_vov.ps1 --status-db 'D:\r2ai_data\crawl\status.duckdb' --scan --batch-size 50 --delay 2
```

Hoặc dùng entry point chung:

```powershell
.\scripts\run_local.ps1 vov --probe
.\scripts\run_local.ps1 vov --scan
# Lượt nhỏ tối đa 10 URL:
.\scripts\run_local.ps1 vov --run --max-urls 10
```

Lệnh `vov` không có cờ nay chỉ lập kế hoạch offline. `--all-vov` là alias của
`--scan`. Các lệnh này đã chuyển sang crawler mới, không gọi cell probe cũ
trong notebook 01 (cell cũ còn dò từ khóa `captcha` trên toàn HTML).

Chạy trực tiếp khi PowerShell chặn `.ps1`:

```powershell
.\.venv\Scripts\python.exe .\crawl\vov.py --scan --status-db 'D:\r2ai_data\crawl\status.duckdb'
```

Chỉ định `--out 'D:\r2ai_data\crawl\vov'` nếu muốn chọn nơi lưu. Mặc định
DB là `R2AI_STATUS_DB`, rồi DB trong `R2AI_DATA_ROOT/crawl` nếu có, cuối cùng
là `status.duckdb` ở repo; data root mặc định `D:\r2ai_data`.
Đóng tiến trình đang ghi DB nguồn trước khi chụp snapshot.

## Resume, tốc độ và dừng

- Một tab Chrome, một URL mỗi lượt, chờ ít nhất 2 giây sau response trước URL
  tiếp theo. Không nghỉ cố định 3 giờ giữa các batch của VOV.
- Bao gồm URL từng `blocked/403` hoặc `blocked/200` trong DB cũ để đánh giá lại
  bằng Chrome. Không reset DB. Bỏ ID đã `ok/non_html/empty` trong DB và 404/410.
- Chạy lại cùng lệnh, cùng DB/output để tiếp tục. Quét và xác thực shard đã
  đóng trước khi bỏ qua ID; không tin riêng một dòng `ok` thiếu raw. Nếu crash
  sau khi đóng shard nhưng chưa ghi đủ CSV, status được khôi phục từ raw.
- Gặp 401/403/429/503 hoặc trang xác minh thật thì dừng ngay, giữ cooldown
  ít nhất 10 phút, hoặc lâu hơn theo `Retry-After`. Chạy lại chỉ tiếp tục khi hết
  cooldown. Không xóa `cooldown.json` để thử liên tục.
- Timeout, redirect ra ngoài VOV, hoặc phản hồi HTML rỗng thì dừng lượt;
  URL 404/410 được bỏ qua ở lần resume. URL chưa thử không nhận status giả.
- Không chạy hai job VOV đồng thời; lock chung nằm ở `crawl/locks/vov.lock`
  khi dùng output mặc định. Nếu chọn output tùy biến, dùng chung thư mục cha
  để giữ cùng domain lock.

```powershell
# Dừng mềm, chờ job lưu xong trước khi đóng terminal:
New-Item -ItemType File -Path 'D:\r2ai_data\crawl\vov\STOP' -Force
# Khi job đã thoát và muốn resume:
Remove-Item -LiteralPath 'D:\r2ai_data\crawl\vov\STOP'
```

`run_local.ps1 stop` cũng tạo STOP cho VOV ở output mặc định. Nếu dùng
`--out` tùy biến, tạo STOP trong đúng thư mục đó.

## Output

```text
D:\r2ai_data\crawl\vov\
  html\home_shard_00000.jsonl.zst
  status.csv
  cooldown.json
  run_summary.json
  crawl.log
  inspection\<id>.html       # chỉ xuất khi gặp HTTP 200 chưa nhận diện được
  inspection\<id>.json
```

Shard giữ sáu trường `id, final_url, http_code, content_type, encoding, body_b64`.
`body_b64` là bytes phản hồi HTTP gốc từ Chrome, không phải DOM đã render hay
Markdown. Ghi và fsync shard, đổi tên thành công rồi mới ghi status tương ứng.
DB nguồn chỉ được đọc qua bản snapshot; dữ liệu mới được lưu độc lập.

Để nhập dữ liệu đã crawl vào pipeline, chạy bước merge riêng khi DB không có
writer khác (xem và kiểm tra dry-run trước):

```powershell
.\.venv\Scripts\python.exe .\crawl\merge_browser_output.py --out 'D:\r2ai_data\crawl\vov' --html-dir 'D:\r2ai_data\crawl\html' --status-db 'D:\r2ai_data\crawl\status.duckdb' --dry-run
# Bỏ --dry-run để thực hiện hợp nhất.
```

Môi trường dùng các dependency có sẵn trong `requirements-local.txt`:
Playwright, BeautifulSoup4, DuckDB, Zstandard và
charset-normalizer. Chrome cần được cài trên Windows; có thể dùng
`--channel msedge` với Edge đã cài.
