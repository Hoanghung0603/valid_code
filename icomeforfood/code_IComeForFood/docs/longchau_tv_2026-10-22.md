# Long Châu theo review 22/10

Review mới giao crawl `crawl/longchau_tv_crawl_odd.csv`: 5.890 ID lẻ theo thứ tự ưu tiên. Những URL còn lại được phía review xử lý bằng Common Crawl. File này không thay thế dữ liệu đã crawl; lệnh bên dưới tiếp tục trên cùng output và bỏ các ID đã lưu thành công ở đó.

Chạy từ mạng nhà, một worker. `--pauses` không kèm số phút nghĩa là **dừng domain ngay ở response 403/429/503 hoặc trang xác minh đầu tiên**. Không xóa `status.csv` hay shard khi chạy lại.
Chỉ chạy `finalize_longchau.py` sau khi crawler kết thúc và `crawl_home_tv/status.csv` đã tồn tại. Nếu dừng ở request đầu tiên, phiên này không có shard HTML mới; bước finalize chỉ có thể đồng bộ trạng thái `blocked`.

Thêm `--plan` vào lệnh dưới để xem số URL còn lại mà không gửi request hoặc sửa output.

```powershell
.\.venv\Scripts\python.exe -u -X utf8 .\crawl\crawl_home.py `
  --corpus .\crawl\longchau_tv_crawl_odd.csv `
  --status-db 'D:\r2ai_data\crawl\status.duckdb' `
  --out 'D:\r2ai_data\crawl\crawl_home_tv' `
  --rate nhathuoclongchau.com.vn=0.25 `
  --ramp 0.05 0.025 20 --workers 1 --pauses --shard-prefix tv `
  --skip-ids 'D:\r2ai_data\crawl\longchau_odd\status.csv' 'D:\r2ai_data\crawl\longchau_odd_fast\status.csv'
```

`--skip-ids` chỉ bỏ ID `ok` đã có shard khi đọc `status.csv`; ID đã bị chặn hoặc lỗi vẫn nằm trong danh sách. Có thể bỏ tùy chọn này nếu hai file status cũ không có ở máy chạy. `--status-db` chụp DB dưới khóa dùng chung với `retry_crawl4ai.py`, nên hai crawler có thể chạy cùng lúc mà Long Châu không gửi lại URL đã có bản lưu trong DB.

Theo dõi 5.890 ID (gồm các bản đã lưu ở crawler cũ, TV và DB chính):

```powershell
.\.venv\Scripts\python.exe -u -X utf8 .\crawl\longchau_progress.py --watch 30
```

Theo dõi riêng phiên chạy mới nhất và tốc độ trang/giờ:

```powershell
.\.venv\Scripts\python.exe -u -X utf8 .\crawl\longchau_progress.py --session-only --watch 30
```

Khi hoàn tất, đưa `crawl_home_tv/html/tv_shard_*.jsonl.zst` và `crawl_home_tv/status.csv` lên thư mục Drive `crawl/html_home_tv/` như review yêu cầu. Các shard chỉ chứa body HTTP gốc với sáu trường của crawler chính. Không tự lấy lại phần đã được giao cho Common Crawl.

Các việc 120ask, PubMed, trích shard v26 và vov.vn vẫn theo review 21/10; danh sách này chỉ đổi phạm vi Long Châu.
