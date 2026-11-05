# Long Châu local: chỉ ID số lẻ có trong DuckDB

Theo review 22/10, dùng [danh sách 5.890 ID ưu tiên](longchau_tv_2026-10-22.md).
Các lệnh bên dưới dùng cú pháp cũ; CLI hiện mặc định
lọc theo `crawl/longchau_tv_crawl_odd.csv`.

Nếu dùng Crawl4AI, xem [longchau_crawl4ai.md](longchau_crawl4ai.md). Các lệnh
không có `--engine crawl4ai` bên dưới dùng HTTPX.

Script kiểm tra `id` và `url` của file ưu tiên với bảng `crawl_status` trong `--status-db`, chỉ giữ
`id % 2 == 1` và domain `nhathuoclongchau.com.vn` (kể cả `www`).
Đây là ID trong bảng, không phải số cuối URL hoặc vị trí trong danh sách xếp hạng.
Không cần corpus Parquet; mặc định là `crawl/longchau_tv_crawl_odd.csv`.

## Chạy trên PowerShell

Đóng tiến trình local đang ghi DB nguồn trước khi chụp snapshot.

```powershell
cd 'D:\New folder (3)\r2aiIComeForFood26'

# Quét ID lẻ chưa có nội dung, gồm ID từng blocked khi cooldown đã hết:
.\scripts\run_local.ps1 longchau-odd --status-db '.\status.duckdb' --scan --retry-blocked
```

Nếu file thực tế tên `statusduck.db`, thay `--status-db '.\status.duckdb'`
bằng `--status-db '.\statusduck.db'`. Đường dẫn có thể là một file DuckDB ở vị trí khác.
Không đổi tên DB hoặc tạo DB rỗng. Script báo đường dẫn nguồn và số ID đủ điều kiện khi bắt đầu.

Nếu PowerShell không cho chạy `.ps1`, dùng trực tiếp:

```powershell
.\.venv\Scripts\python.exe .\crawl\longchau_odd.py --status-db '.\status.duckdb' --scan --retry-blocked
```

Môi trường cần `duckdb`, `httpx`, `zstandard`, `charset-normalizer`. Khi chưa cài:

```powershell
.\.venv\Scripts\python.exe -m pip install duckdb httpx zstandard charset-normalizer
```

## Cách chọn và lưu

- Chỉ request ID số lẻ thuộc Long Châu và tồn tại trong DB nguồn.
- Bỏ ID đã có raw ở output, hoặc đã `ok`/`non_html`/`empty` trong DB.
- Bỏ 404/410; mặc định bỏ cả blocked/401/403/429. `--retry-blocked` cho phép
  đưa ID từng bị chặn vào hàng đợi, nhưng vẫn giữ cooldown của domain.
- `--scan` không yêu cầu một bước probe riêng; chỉ tiếp tục khi hết cooldown.
- Một request mỗi 30 giây; mỗi 50 URL lưu shard rồi nghỉ 3 giờ, tự tiếp tục trong cùng job.
  Có thể chậm hơn với `--delay 60` hoặc giảm batch bằng `--batch-size 10`.
- Gặp 401/403/429/503 hoặc trang challenge: dừng domain, lưu cooldown ít nhất 6 giờ
  (hoặc theo `Retry-After` dài hơn). Không tự thử tiếp khi bị chặn.
- DB nguồn chỉ được đọc qua snapshot, không reset status và không cập nhật trực tiếp.
  Kết quả mới lưu riêng, sau đó dùng bước `merge-home` nếu muốn nhập lại vào pipeline.

Output mặc định:

```text
D:\r2ai_data\crawl\longchau_odd\
  html\home_shard_*.jsonl.zst  # raw HTML bytes, nén Zstandard
  status.csv                   # status của mỗi URL đã thử
  cooldown.json                # trạng thái nghỉ/dừng domain
  run_summary.json             # tổng kết lượt chạy
  crawl.log
```

Chạy lại cùng lệnh để tiếp tục; ID đã lưu raw sẽ được bỏ qua.
Thêm file `STOP` vào thư mục output để dừng mềm:

```powershell
New-Item -ItemType File -Path 'D:\r2ai_data\crawl\longchau_odd\STOP' -Force
```

Sau khi tiến trình đã thoát, xóa riêng `STOP` khi muốn chạy tiếp. Không xóa `cooldown.json`.
Lệnh không có `--scan`, `--probe`, `--run` chỉ hiển thị kế hoạch offline, không gửi HTTP.
