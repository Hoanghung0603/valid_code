# Long Châu: Bản Crawl4AI Nhanh và An Toàn (Bản Riêng Biệt)

Review 22/10 thu hẹp phần cần crawl còn 5.890 ID. CLI bản nhanh hiện mặc định
`crawl/longchau_tv_crawl_odd.csv`; xem [lệnh TV được review yêu cầu](longchau_tv_2026-10-22.md)
trước khi chạy. Bộ nhanh vẫn là phương án riêng và cần `--retry-blocked` nếu
muốn đưa ID từng bị chặn trở lại hàng đợi sau cooldown.
DB mặc định là `D:\r2ai_data\crawl\status.duckdb`, đúng bản được dùng để đồng bộ
trạng thái khi crawl; `--corpus` có thể truyền tường minh để kiểm tra đúng CSV.
CSV chỉ được dùng khi từng ID lẻ và URL Long Châu khớp với DB. Lệnh không đưa
ID Long Châu khác ngoài CSV vào hàng đợi.
Nó cũng tôn trọng 10 phút cooldown sau lần chặn gần nhất trong
`crawl_home_tv/status.csv` khi cùng chạy trên máy này. Shard được ghi xong trước
khi trạng thái tương ứng được đẩy vào DuckDB.

Bản crawl này được xây dựng độc lập, **giữ nguyên 100% code cũ** (`crawl/longchau_odd.py` và `crawl/longchau_crawl4ai.py`) để bạn có thể chạy thử nghiệm, trích xuất (extract) và so sánh trực tiếp trước khi quyết định.

---

## 1. Tóm tắt các điểm tối ưu (Hiệu suất & An toàn)

| Hạng mục | Bản Cũ (`longchau_crawl4ai.py`) | Bản Nhanh Mới (`longchau_crawl4ai_fast.py`) | Lợi ích |
|---|---|---|---|
| **Tải ảnh & font** | Tải đầy đủ toàn bộ ảnh, banner, webfont | `text_mode=True` (chặn ảnh & font ở mức Chromium C++) | Giảm **70–80% băng thông**, tải trang nhanh gấp **1.5x–2x** |
| **Xử lý CPU Python** | Crawl4AI tự parse Markdown, trích xuất bảng, chunking rồi vứt bỏ | Bật `prefetch=True` (ngắt mạch short-circuit) | Giảm **100–300ms CPU** mỗi trang, không tốn RAM tính Markdown thừa |
| **Quản lý RAM** | Không giới hạn | `memory_saving_mode=True`, recycle tab sau 100 trang | Không bao giờ bị tràn RAM khi crawl qua đêm |
| **Nhịp request** | Delay cố định | Tăng từ 0,05 lên tối đa 0,3 request/giây theo thời gian; jitter chỉ cộng thêm thời gian chờ | Giảm tải khi bắt đầu mỗi lượt |
| **Khi bị chặn** | Dừng lượt | Chỉ ID lẻ, `check_robots_txt=True`, dừng ngay khi 403/challenge, cooldown tối thiểu 10 phút | Tôn trọng `Retry-After` nếu lâu hơn |

---

## 2. So sánh chất lượng trích xuất (Extraction Parity)

Kết quả chạy trích xuất thực tế qua bộ lọc `extract_html_doc` + `postfilter v1.12`:

| ID bài viết | Tiêu đề | Ngày đăng | Bản Cũ (Blocks / Chars) | Bản Nhanh (Blocks / Chars) | Độ khớp nội dung |
|---|---|---|---|---|---|
| **258903** | Phun môi bị nổi mụn trắng... | 2022-11-29 | 59 blocks / 7.958 ký tự | 59 blocks / 7.958 ký tự | **Khớp 100% từng chữ** |
| **232965** | Chỉ số AMH thấp có làm IVF được không? | 2023-10-04 | 41 blocks / 6.543 ký tự | 41 blocks / 6.543 ký tự | **Khớp 100% từng chữ** |
| **263557** | Sữa công thức pha để được bao lâu? | 2023-12-11 | 53 blocks / 6.875 ký tự | 47 blocks / 6.780 ký tự | **Khớp sạch** (lược bớt popup cookie) |

*Thời gian load trang:* Giảm từ **~2.3–2.6s** xuống còn **~1.1–1.5s/trang**.

---

## 3. Nhịp tăng tốc

Mỗi lần khởi động tiến trình, tốc độ bắt đầu lại từ 0,05 request/giây. Đồng hồ
tăng tốc chỉ bắt đầu ngay trước URL đầu tiên, sau khi mở trình duyệt và hoàn tất
thời gian chờ giữa hai lượt. Cứ 20 phút tăng thêm 0,025 request/giây, dừng ở
0,3 request/giây sau 200 phút:

| Thời gian chạy | Trần request/giây | Chờ tối thiểu sau mỗi response |
|---|---:|---:|
| 0–20 phút | 0,05 | 20 giây |
| 20–40 phút | 0,075 | 13,33 giây |
| 40–60 phút | 0,10 | 10 giây |
| 60–80 phút | 0,125 | 8 giây |
| 80–100 phút | 0,15 | 6,67 giây |
| 100–120 phút | 0,175 | 5,71 giây |
| 120–140 phút | 0,20 | 5 giây |
| 140–160 phút | 0,225 | 4,44 giây |
| 160–180 phút | 0,25 | 4 giây |
| 180–200 phút | 0,275 | 3,64 giây |
| Từ 200 phút | 0,30 | 3,33 giây |

Để đạt trần 0,3, bỏ `--delay 5` khỏi lệnh cũ: tùy chọn đó giữ mức chờ tối thiểu
5 giây và giới hạn tốc độ thực tế ở 0,2 request/giây. Mặc định là 3,33 giây;
jitter chỉ cộng thêm tối đa 8% thời gian chờ. Tốc độ thực tế có thể thấp hơn do thời gian tải
trang. Lịch này không bảo đảm Cloudflare sẽ không chặn. Nếu gặp 403/challenge,
crawler dừng và giữ cooldown; lượt chạy lại bắt đầu từ 0,05 request/giây.

## 4. Lệnh chạy thử nghiệm & so sánh

### Bước 1: So sánh trực tiếp chất lượng trích xuất (Benchmark)
Chạy script so sánh song song 3–5 URL trên terminal:

```powershell
.\.venv\Scripts\python.exe scripts/compare_extract_longchau.py --benchmark 3
```

Hoặc kiểm tra độ trích xuất từ shard cũ:
```powershell
.\.venv\Scripts\python.exe scripts/compare_extract_longchau.py --old-shard D:/r2ai_data/crawl/longchau_odd/html/tv_shard_00000.jsonl.zst
```

### Bước 2: Chạy thử bản nhanh (Thử nghiệm 20 URL đầu)
Bản mới lưu dữ liệu độc lập vào `D:\r2ai_data\crawl\longchau_odd_fast`:

```powershell
.\.venv\Scripts\python.exe crawl/longchau_odd_fast.py --scan --retry-blocked --trial-urls 20
```

### Bước 3: Quét liên tục (Chạy dài qua đêm)
```powershell
.\.venv\Scripts\python.exe crawl/longchau_odd_fast.py --corpus .\crawl\longchau_tv_crawl_odd.csv --status-db 'D:\r2ai_data\crawl\status.duckdb' --scan --retry-blocked --continuous --batch-size 50
```

> **Lưu ý:**
> - Bản fast dùng output và lock riêng. Không chạy song song hai crawler Long Châu trên cùng IP.
> - Nếu gặp trang xác minh hoặc 403, tiến trình dừng và kích hoạt cooldown tối thiểu 10 phút.
