# Chạy local theo review 20/10 — build v26

Phạm vi Long Châu đã đổi theo [review 22/10](longchau_tv_2026-10-22.md):
chỉ crawl 5.890 ID trong `crawl/longchau_tv_crawl_odd.csv`.

Repo: `D:\New folder (3)\r2aiIComeForFood26`. Dữ liệu mặc định: `D:\r2ai_data`.
Các lệnh dùng đúng cell trong notebook 01/03, không có một bản cleaner thứ hai.

## Chuẩn bị

```powershell
cd 'D:\New folder (3)\r2aiIComeForFood26'
.\scripts\run_local.ps1 install
.\scripts\run_local.ps1 prepare
.\scripts\run_local.ps1 preflight
.\scripts\run_local.ps1 plan
```

`prepare` tạo các thư mục dữ liệu và chép hints/domain plan/manifest từ repo nếu chưa có.
DB mặc định là `D:\r2ai_data\crawl\status.duckdb` nếu đã tồn tại; nếu chưa có thì dùng
`status.duckdb` ở gốc repo để giữ tiến độ hiện có. Có thể chỉ định `--status-db 'đường dẫn'`
và `--data-root 'đường dẫn'` cho mọi lệnh. Không khởi tạo DB rỗng thay thế DB đã crawl.

`plan` chọn hàng đợi trên bản sao DB, không gửi request và không sửa DB gốc. Xem `reports/local_crawl_plan.csv`.

## Thu thập

```powershell
.\scripts\run_local.ps1 crawl --workers 32
.\scripts\run_local.ps1 status
.\scripts\run_local.ps1 pubmed
```

Crawl mặc định full, resume từ status DB, chung ngân sách 1 request/giây cho toàn bộ 39.net,
và dùng HTTP/1.1. Notebook kiểm phiên bản hàm để tránh chạy cell Colab cũ. Checkpoint 60 giây:
shard đóng trước, rồi status và metrics. Nếu chưa đồng bộ HTML cũ về máy, shard mới luôn bắt
đầu sau số shard lớn nhất trong DB để tránh trùng số lúc hợp nhất với Drive.

PubMed dùng Europe PMC `searchPOST`, tối đa 1.000 PMID/request, tái sử dụng cache. Dừng/resume
không làm mất batch đã ghi. Xem `pubmed_coverage.csv` và `pubmed_missing_pmids.csv`.

Long Châu, Tâm Anh, Pharmacity và zysjonline được bỏ khỏi crawler thường; 120ask thu trên VPS.
Hai domain iiyi chỉ probe tối đa một lần/domain/24 giờ; server hồi phục mới xếp lại URL 521.
Hết queue chưa đồng nghĩa hoàn tất 1,2 triệu URL: xem báo cáo status theo domain.

VOV dùng crawler Chrome headless riêng theo quy trình checkpoint Long Châu.
Xem [hướng dẫn VOV](vov_local.md). Lập kế hoạch, thử một URL rồi crawl tiếp:

```powershell
.\scripts\run_local.ps1 vov
.\scripts\run_local.ps1 vov --probe
.\scripts\run_local.ps1 vov --scan
```

Dùng Chrome đã cài trên Windows. Nếu không khởi chạy được Chrome, cài browser bằng
`.\.venv\Scripts\python.exe -m playwright install chrome`.
Gặp chặn thì dừng ngay và lưu cooldown; không thử hết hàng đợi.
Raw và status lưu riêng ở `D:\r2ai_data\crawl\vov`, DB nguồn chỉ được đọc.
Chữ `captcha` trong CSS/script không còn làm bài hợp lệ bị đánh nhầm `blocked`.

## Hợp nhất home và trích full

Đồng bộ `crawl/html_home/status.csv` và `home_shard_*.jsonl.zst` từ NT rồi chạy:

```powershell
.\scripts\run_local.ps1 merge-home
```

Đồng bộ các shard cũ từ Drive/VPS vào `D:\r2ai_data\crawl\html` và status DB tương ứng.
Không thể trích lại dữ liệu chỉ từ file DB: DB không chứa HTML. Ba ZIP Gold đặt ở gốc repo
hoặc data root; PubMed Gold có thể cần API/cache cho 8 PMID.

```powershell
.\scripts\run_local.ps1 extract
```

Lệnh trích chạy cleaner selftest → cả ba Gold → `check_gate` → parity → full v26,
chọn `latest-shard-http200-v1`, giữ `db_status_mismatch`, giữ placeholder cho ID chưa có HTML.
Mỗi run có thư mục report riêng; không tìm/xóa các thư mục `latest` trùng tên trên Drive.
Đầu ra: `docs/full_<RUN_ID>/`, báo cáo `full_extraction_summary`, `full_missing`,
`full_db_status_mismatch`, `full_domain_quality`, `full_duplicate_text_groups`, `full_offset_errors`.

## Dừng

Chạy ở terminal thứ hai:

```powershell
.\scripts\run_local.ps1 stop
```

Đợi terminal chạy lưu xong rồi mới đóng. Xóa `D:\r2ai_data\crawl\STOP` khi chủ ý bắt đầu lượt mới.
Trên Windows, đóng kernel/connection đang ghi DB trước khi lấy snapshot bằng một tiến trình khác
hoặc chạy extraction. Lệnh `status` dùng snapshot chỉ đọc và không sửa crawl_status.


## Long Châu: chỉ ID lẻ, dữ liệu một phần — cập nhật 21/10

ID và URL nay lấy **trực tiếp từ bảng `crawl_status` trong DB**, không từ corpus Parquet.
Đọc `docs/longchau_db_local.md`. Dùng `--status-db` để chọn chính xác file nguồn:

```powershell
.\scripts\run_local.ps1 longchau-odd --status-db '.\status.duckdb' --scan --retry-blocked
```

`--retry-blocked` bao gồm ID từng bị chặn sau cooldown; gặp chặn lần nữa vẫn dừng ngay.
Nếu không muốn thử lại blocked, bỏ cờ này. DB nguồn không bị sửa.

NT báo chỉ thu được 3.472/79.598 trang; không coi Long Châu đã đủ. Yêu cầu hiện tại chỉ cho
phép luồng riêng lấy ID gốc lẻ (`id % 2 == 1`). Crawler thường vẫn loại Long Châu; lệnh
`crawl_home.py` / `crawl_browser.py` crawl toàn domain đã được khóa cho Long Châu.

```powershell
.\scripts\run_local.ps1 longchau-odd          # chỉ lập danh sách, không gửi request
.\scripts\run_local.ps1 longchau-odd --probe  # tối đa một request sau cooldown
# Chỉ chạy khi probe đã lấy được HTML 200:
.\scripts\run_local.ps1 longchau-odd --run --batch-size 50 --delay 30
```

Một luồng, ít nhất 30 giây giữa các request, tối đa 50 trang/lượt rồi nghỉ 3 giờ. Mức này
không bảo đảm tránh chặn. Không có VPN, proxy, đổi IP, xử lý CAPTCHA hay giả lập client.
Gặp 403/429/503 hoặc trang xác minh thì dừng ngay, lưu cooldown tối thiểu 6 giờ (tôn trọng
Retry-After dài hơn); hết cooldown chỉ cho probe một request, không tự chạy lại hàng đợi.
Không reset blocked về pending, không sửa DB nguồn. ID đã ok/raw, blocked hoặc 404/410 được bỏ qua.

HTML bytes gốc lưu trong `D:\r2ai_data\crawl\longchau_odd\html\home_shard_*.jsonl.zst`,
đúng 6 trường của pipeline. `status.csv`, `crawl.log`, `cooldown.json`, `run_summary.json` cùng
thư mục; summary ghi `partial_domain=true`. Có khóa chống chạy hai instance của luồng này.
Dừng mềm bằng file `STOP` trong thư mục output hoặc `scripts/run_local.ps1 stop`.
Để chạy lại có chủ ý, xóa STOP sau khi tiến trình cũ đã thoát; cooldown vẫn được giữ.

Hợp nhất phần raw đã thu được:

```powershell
.\scripts\run_local.ps1 merge-home --home-dir D:\r2ai_data\crawl\longchau_odd
```


## Quét ID lẻ không cần bước probe riêng

Sau khi hết cooldown, dùng một lệnh:

```powershell
cd "D:\New folder (3)\r2aiIComeForFood26"
.\scripts\run_local.ps1 longchau-odd --scan --batch-size 50 --delay 30
```

Chế độ `--scan` tự duyệt các ID lẻ còn đủ điều kiện, lưu HTML 200 và ghi status từng URL.
Không cần chạy `--probe` hoặc `--run` trước. Mỗi request cách nhau ít nhất 30 giây;
cứ 50 URL thì ghi shard, nghỉ 3 giờ rồi tự tiếp tục trong cùng tiến trình.
Giữ cửa sổ terminal mở; file STOP vẫn dừng được cả trong lúc nghỉ.

Nếu gặp 403/429/503 hoặc trang xác minh, dừng toàn domain ngay và giữ cooldown ít nhất 6 giờ.
Không bỏ qua cooldown để quét tiếp; URL chưa thử không được kết luận là lỗi hoặc không lấy được.
Sau cooldown, chạy lại `--scan` để tiếp tục, các ID đã có raw được bỏ qua.

Quy trình `--probe` rồi `--run` bên dưới là lựa chọn thủ công; không chạy đồng thời với `--scan`.

