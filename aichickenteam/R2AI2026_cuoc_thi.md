# R2AI2026 – Truy hồi tài liệu y khoa đa ngôn ngữ (Shared Task 3)

- **Link:** https://leaderboard.aiguru.com.vn/competitions/15/
- **Nền tảng:** Codabench (AI Guru), Docker image `codalab/codalab-legacy:py37`
- **Tổ chức bởi (tài khoản):** BM25 Baseline
- **Trạng thái lúc xem (23/09/2026):** chưa có giai đoạn nào hoạt động · 25 người tham gia · 0 bài nộp · đăng ký cần BTC duyệt (không tự động)

---

## 1. Bối cảnh bài toán

Tri thức y sinh phân bố không đều giữa các ngôn ngữ: nguồn tiếng Việt còn hạn chế về chiều sâu, trong khi tiếng Anh và tiếng Trung có kho tài nguyên rất lớn. Điều này tạo ra "khoảng cách thông tin y tế" với người dùng tiếng Việt.

**Nhiệm vụ:** Cho tập truy vấn tiếng Việt Q = {q₁, …, qₙ} và các nguồn dữ liệu y sinh đa ngôn ngữ (Việt, Anh, Trung). Các đội tự xây dựng pipeline: thu thập → tiền xử lý → phân đoạn (chunking) → lập chỉ mục → truy xuất. Với mỗi qᵢ, hệ thống tìm và xếp hạng tài liệu/đoạn nội dung liên quan, bất kể ngôn ngữ nguồn.

Đánh giá ở **2 cấp**: tài liệu (document) và đoạn nội dung (chunk).

## 2. Mục tiêu

1. **Truy hồi chính xác cấp tài liệu:** tìm đúng tài liệu liên quan; truy hồi xuyên ngôn ngữ Việt–Anh–Trung, không chỉ so khớp từ khoá.
2. **Truy hồi chính xác nội dung:** tìm đúng các đoạn chứa thông tin y khoa liên quan; cách chunking do đội tự thiết kế.
3. **Hiểu truy vấn y khoa tiếng Việt:** thuật ngữ y khoa/lâm sàng, truy vấn có nhiều ý trong một câu.

## 3. Mốc thời gian

| Ngày | Sự kiện |
|---|---|
| 01/10/2026 | Phát hành tập public test |
| 31/10/2026 | Hạn chót nộp bài public test |
| 01/11/2026 | Bắt đầu private test |
| 04/11/2026 | Hạn chót nộp hệ thống |
| 11/11/2026 | Công bố kết quả chung cuộc |

> Mọi hạn chót là **23:59 giờ Việt Nam (UTC+7)** theo trang Overview.

**Các giai đoạn trên hệ thống:**

| Giai đoạn | Bắt đầu | Kết thúc | Số bài/ngày | Giới hạn thời gian chạy |
|---|---|---|---|---|
| Kiểm thử công khai | 01/10/2026 07:00 (GMT+7) | 31/10/2026 07:00 (GMT+7) | 10 | 600s |
| Kiểm thử riêng | 01/11/2026 07:00 (GMT+7) | 04/11/2026 07:00 (GMT+7) | 5 | 600s |

⚠️ Giờ kết thúc cài trên hệ thống (07:00) sớm hơn mốc 23:59 ghi trong Overview. Nên nộp trước 07:00 cho chắc.

## 4. Quy định về dữ liệu ngoài & mô hình

- Được dùng dữ liệu ngoài, nhưng **phải trích dẫn rõ nguồn** để BTC kiểm tra.
- Chỉ được dùng PLM/LLM **mã nguồn mở hoặc công khai trọng số** (ví dụ Hugging Face). **Cấm LLM đóng** (GPT-4o, Gemini, …).
- Mô hình phải **phát hành trước 01/06/2026** và có kích thước **≤ 14B tham số**.

## 5. Dữ liệu

**BTC cung cấp:**
- **Danh sách nguồn:** URL các bài viết y khoa tiếng Việt và tiếng Trung (JSONL). Đội tự crawl, xử lý và xây dựng cơ sở tri thức.
- **Tiếng Anh:** không có danh sách sẵn. Đội tự tìm trên **PubMed** qua NCBI Entrez E-utilities hoặc Europe PMC.
- **Test set:** câu hỏi tiếng Việt (JSONL).

BTC **không** cung cấp sẵn embedding/indexing. Đội tự chọn embedding đa ngôn ngữ, chiến lược index/search, reranking, hậu xử lý và dữ liệu y học mở bổ trợ.

**Định dạng truy vấn (JSONL):**
```json
{"id": 1, "query": "Cần làm gì đối với tình trạng tắc nghẽn đường tiết niệu do sỏi thận?"}
```

**Định dạng nguồn Việt/Trung (JSONL):**
```json
{"id": 1, "url": "https://example.com/article"}
```

**Ví dụ lấy ứng viên PubMed:**
```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=kidney+stone+urinary+obstruction&retmax=100&retmode=json
https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=kidney%20stone%20urinary%20obstruction&format=json&pageSize=100
```

## 6. Đánh giá

Các chỉ số được tính **cho từng truy vấn**, sau đó lấy trung bình trên toàn bộ truy vấn (**macro-average**), ở cả cấp document và cấp chunk:

- **Precision** = số kết quả đúng / tổng số kết quả trả về
- **Recall** = số kết quả đúng / tổng số kết quả liên quan theo nhãn
- **F2** = (5 × P × R) / (4 × P + R). F2 coi trọng Recall hơn Precision.
- **F2-macro** = trung bình F2 của mọi truy vấn.

**Cột trên bảng xếp hạng (RESULTS):** FINAL SCORE, EXECUTION ACCURACY, TABLES F2-MACRO, DOCS F2-MACRO, TABLES PRECISION, TABLES RECALL, TABLES MRR5, DOCS PRECISION, DOCS RECALL, DOCS MRR5, ANSWER ACCURACY.

> ⚠️ Tên các cột này (TABLES, EXECUTION ACCURACY, ANSWER ACCURACY) và comment HTML của trang Evaluation ("Financial Table Retrieval and Text-to-Pandas") có vẻ được sao chép từ một cuộc thi khác. Nhiều khả năng "TABLES" tương ứng với cấp chunk. Nên hỏi BTC cách tính FINAL SCORE.

## 7. Nộp bài

**File `.json`** (một mảng), phải có đủ **toàn bộ** câu hỏi:
```json
[
  {
    "id": 1,
    "relevant_docs": ["doc_id1", "doc_id2"],
    "relevant_chunks": [
      {"doc_id": "doc_id1", "chunk_text": "Đoạn trích nguyên văn từ doc_id1."},
      {"doc_id": "doc_id2", "chunk_text": "Đoạn trích nguyên văn từ doc_id2."}
    ]
  }
]
```

- `id`: id câu hỏi (integer).
- `doc_id`: với Việt/Trung là `id` trong file nguồn của BTC; với tiếng Anh là **PMID** của PubMed.
- `chunk_text`: **trích nguyên văn** từ tài liệu gốc, không được viết lại hay sinh mới.
- Nếu không có dự đoán ở một cấp, vẫn phải có trường đó với **danh sách rỗng**.
- Nén thành **ZIP chứa đúng 1 file ở gốc** (không có thư mục con), rồi upload ở mục *My Submissions*.
- Bài thiếu file hoặc thiếu câu sẽ không được chấm và **không bị tính** vào số lượt nộp.

**Quy định nộp:**
1. Tối đa **10 bài/ngày**.
2. Vòng Private: tối đa **5 bài tổng cộng** mỗi người dùng. Riêng hệ thống đang cài 5 bài/ngày.
3. Mỗi đội dùng **một tên người dùng đại diện**.
4. Kết quả chỉ chính thức khi đội nộp **working notes paper** mô tả đầy đủ phương pháp.
5. BTC có quyền loại bài không tuân thủ.

## 8. Điều khoản chính

- BTC có quyền chấm dứt, sửa đổi hoặc tạm ngưng cuộc thi.
- Đồng ý cho công bố điểm tại hội thảo và kỷ yếu. BTC quyết định chỉ số và điểm cuối cùng.
- Phải tuân thủ Điều khoản Tham gia và Thỏa thuận Sử dụng Dữ liệu (gửi qua email), không vi phạm bản quyền/sở hữu trí tuệ và luật về dữ liệu.
- BTC được quyền sử dụng bài nộp thắng giải cùng mã nguồn và dữ liệu liên quan cho mọi mục đích.
- **Mỗi người 1 tài khoản.** Được lập đội, nhưng mỗi người **chỉ ở 1 đội**, và **mỗi đội chỉ 1 tài khoản** được nộp bài.
- BTC có thể loại thí sinh gian lận.

## 9. Ban tổ chức & liên hệ

**AI Guru – Công ty Cổ phần Tập đoàn Dagoras Group**
Tầng 8, số 80 Duy Tân, Cầu Giấy, Hà Nội

- Nguyễn Thị Minh Nguyệt – 0981544974 – nguyetntm@dagoras.io
- Vũ Thị Thuỳ Linh – 0961891198 – linhvtt@dagoras.io
