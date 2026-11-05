# R2AI 2026 – ViBioMIR

Code của đội cho vòng 3 R2AI 2026: truy hồi tài liệu y sinh đa ngôn ngữ (truy vấn tiếng Việt; tài liệu tiếng Việt, Anh, Trung).

## Đọc trước
- `PHAN_CONG.md`: phân công, quy định bắt buộc, schema dữ liệu, deadline.
- `DAC_TA_LAM_SACH_v1.md`: đặc tả làm sạch và trích xuất (phiên bản ghi ở dòng đầu).
- `docs/data_sources.md`: nguồn dữ liệu ngoài (bắt buộc cập nhật khi dùng nguồn mới).

## Dữ liệu crawl đã chia sẻ trên `main`

- [Tải HTML VOV mở trực tiếp](crawl/vov_snapshot_2026-10-30/html_downloads/README.md): **10 ZIP chứa đủ 9.784 file HTML**, giải nén rồi mở `index.html`, không cần chạy code. [Shard gốc và manifest](crawl/vov_snapshot_2026-10-30/README.md).

- [HTML bổ sung 4 domain ngày 27/10](crawl/four_domain_html_snapshot_2026-10-27/README.md): 677 HTML mới; manifest đủ 927 ID của Sức khỏe Cộng đồng, Thanh Niên, Sức khỏe & Đời sống và ask.39.net, gồm 250 bản đã có trong snapshot cũ.
- [Snapshot crawl cập nhật 26/10](crawl/other_crawl_snapshot_2026-10-24/README.md): `status.duckdb.zst`, 374 HTML shard trong `html/` và `manifest.csv`. Long Châu có 5.879/5.890 ID `ok` với HTML đầy đủ; 11 URL trả 404.
- [Trạng thái Long Châu](crawl/other_crawl_snapshot_2026-10-24/longchau_status.csv) và [trạng thái Pharmacity](crawl/other_crawl_snapshot_2026-10-24/pharmacity_status.csv): mỗi URL/ID có trạng thái và cột cho biết tệp HTML có trên máy hay không.
- [Danh sách shard còn thiếu](crawl/other_crawl_snapshot_2026-10-24/missing_shards.csv).

Hãy chọn nhánh **`main`** trên GitHub để thấy các tệp này. GitHub không xem trực tiếp được DuckDB nén; tải `status.duckdb.zst` về và giải nén theo hướng dẫn trong snapshot.

## Quy định
- Repo **private**. Các snapshot crawl ở mục trên đã được đưa lên `main`; secrets (token HF, khóa RunPod, cookie) vẫn chỉ để trong `.env`.
- Không dùng LLM đóng ở bất kỳ bước nào. Chỉ mô hình mở ≤ 15B, phát hành trước 01/08/2026.
- Dữ liệu crawler đang chạy nằm ở `D:\r2ai_data\`; các snapshot đã chốt nằm trong repo theo liên kết ở trên.

## Cài đặt
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Công cụ đánh giá (`eval/`)
| Script | Việc |
|---|---|
| `r2ai_eval.py` | Chuẩn hóa văn bản theo luật BTC, tách token, LCS |
| `score_extraction.py` | Chấm bộ trích trên bộ chuẩn (recall, tỉ lệ rác, khối bắt buộc, heading, cổng duyệt) |
| `score_submission.py` | Chấm file nộp offline (P/R/F2 mức doc và chunk) |
| `gold_tools.py` | Tải HTML, tách khối, ghi bộ chuẩn |
| `make_gold_manifest.py` | Chọn trang cho bộ chuẩn, thống kê domain |

Ví dụ:
```
python eval/score_extraction.py --gold D:/r2ai_data/gold --pred D:/r2ai_data/reports/extract_gold --out D:/r2ai_data/reports/extract_scores
```

## Git
Nhánh `main` luôn chạy được. Mỗi việc một nhánh `tv/<việc>` hoặc `nt/<việc>`, merge qua pull request, NT duyệt.
