# R2AI 2026 – ViBioMIR: Phân công nhiệm vụ nhóm

> Phiên bản: v1 – 14/10/2026. Người soạn: nhóm trưởng.
> Tài liệu này là **nguồn chuẩn** cho việc ai làm gì, khi nào, giao ra cái gì. Có gì chưa rõ thì hỏi nhóm trưởng trước khi làm, đừng tự đoán.

---

## 0. Đọc trước (5 phút)

### 0.1 Bài toán tóm tắt
- Đầu vào: **1.200 truy vấn y khoa bằng tiếng Việt** (`query.parquet`: `id`, `query`).
- Kho tài liệu: `links_corpus_vi_en_zh.parquet` (`id`, `url`) trên Hugging Face `AIGuruTinix/ViBioMIR`. File hiện có **1.220.598 URL** (VI 740.598 · EN 240.000 · ZH 240.000); email BTC ghi 1.194.469, chênh đúng bằng số URL của vinmec.com. **NT chốt: dùng bản 1.220.598 URL (crawl cả vinmec).**
  - Chỉ có **94 domain**; top 20 domain chiếm 85,5% URL. **Toàn bộ tiếng Anh là PubMed.**
  - **Đặc tả làm sạch chi tiết: `DAC_TA_LAM_SACH_v1.md`** (đọc kỹ trước khi làm T1.2).
  - BTC **chỉ đưa URL**, đội phải **tự crawl** nội dung.
- Đầu ra cho mỗi truy vấn: danh sách `id` tài liệu liên quan, và danh sách đoạn văn (chunk) liên quan, trong đó **chunk phải là nguyên văn trích từ tài liệu gốc**.
- Chấm bằng **F2** (coi trọng recall gấp 4 lần precision) ở hai mức: tài liệu và đoạn. Một chunk được tính đúng nếu trùng ≥ 40% (theo dãy token con chung dài nhất) với một đoạn tham chiếu của BTC.
  - Hệ quả cho phần dữ liệu: **văn bản mình lưu phải giống hệt chữ trên trang gốc**. Lưu sai, thiếu hay "sửa đẹp" văn bản đều làm mất điểm trực tiếp.

### 0.2 Mốc thời gian
| Ngày | Mốc |
|---|---|
| 22/10 | Mentor Day 3 (BTC dự kiến) |
| **31/10, 23:59** | **Hết Vòng 1 – hạn nộp public** |
| 01–10/11 | Vòng 2 – mentoring, tối ưu |
| 04/11 | Hạn nộp hệ thống theo đề (đang hỏi BTC xác nhận lại) |
| 11/11 | Demo Day |

### 0.3 Vai trò
| Người | Vai trò | Sở hữu (quyết định cuối) |
|---|---|---|
| **Nhóm trưởng (NT)** | Chiến lược, **quy tắc làm sạch và kiểm soát chất lượng dữ liệu**, **chunking**, retrieval, đánh giá, nộp bài, liên hệ BTC | Thiết kế hệ thống, quy tắc làm sạch, duyệt chất lượng từng domain, toàn bộ chunking, mô hình, cách trộn kết quả, mọi lần nộp |
| **Thành viên B (TV)** | Kỹ sư dữ liệu và hạ tầng | Crawler, **khung trích xuất** và **luật theo từng domain (theo danh sách NT duyệt)**, chạy hàng loạt, báo cáo thống kê, lưu trữ, chạy job theo đặc tả, đóng gói tái lập |

**Nguyên tắc phân quyền:**
1. TV **triển khai theo đặc tả** NT đưa ra. Nếu thấy đặc tả có vấn đề hoặc có ý tưởng cải tiến → ghi vào mục "Đề xuất" trong báo cáo ngày và báo NT; **không tự đổi quy tắc xử lý dữ liệu**.
2. Mọi tham số ảnh hưởng đến điểm (cách chunk, ngưỡng lọc, cách chọn văn bản) phải **để cấu hình được** (file config), không hard-code. NT chốt giá trị cuối.
3. **Chỉ NT nộp bài** lên leaderboard (đề quy định một tài khoản nộp mỗi đội).
4. **Chỉ NT liên hệ BTC.** TV có câu hỏi thì gửi NT để gom lại.

---

## 1. Quy định bắt buộc (vi phạm có thể bị loại)

1. **Không dùng LLM đóng** (GPT, Gemini, Claude API…) ở **bất kỳ bước nào của pipeline** (làm sạch, dịch, gán nhãn, sinh dữ liệu). Chỉ dùng mô hình có trọng số công khai, **≤ 15B tham số**, **phát hành trước 01/08/2026**. Trước khi dùng mô hình mới nào, báo NT để kiểm tra ngày phát hành và license.
2. **Dữ liệu ngoài** (từ điển, bộ dữ liệu y khoa…) phải **ghi rõ nguồn** (URL, license, ngày tải) vào `docs/data_sources.md`.
3. **Không công khai dữ liệu crawl** (không push lên GitHub public, không đăng dataset public). Repo và dataset trên Hugging Face đều để **private**.
4. **Không commit secrets** (token Hugging Face, khóa RunPod, cookie…). Dùng file `.env` và đưa vào `.gitignore`.
5. Crawl lịch sự: tôn trọng giới hạn tốc độ của từng domain, không tấn công dồn dập một trang.

---

## 2. Hạ tầng và quy ước chung

### 2.1 Thư mục
- **Code:** Git repo private https://github.com/namgau/r2aiIComeForFood26 (NT mời TV). Không để dữ liệu trong repo.
- **Dữ liệu trên máy:** đặt ngoài OneDrive, ví dụ `D:\r2ai_data\` (máy NT) hoặc thư mục tương đương trên máy TV.

```
r2ai_data/
├── raw/                      # input BTC (parquet gốc, không sửa)
├── crawl/
│   ├── status.duckdb         # trạng thái từng URL
│   └── html/shard_XXXXX.jsonl.zst   # HTML thô nén, 10.000 id/shard
├── docs/docs_XXXXX.parquet   # văn bản đã trích + metadata
├── chunks/<chunk_config>/chunks_XXXXX.parquet
├── reports/                  # báo cáo thống kê, mẫu kiểm tra
└── tmp/
```

- **Kho chung:** dataset **private** trên Hugging Face Hub (NT tạo). Chỉ đẩy `docs/` và `chunks/` (Parquet), không đẩy HTML thô.
- **GPU:** RunPod (NT quản lý tài khoản và thanh toán). TV được cấp quyền chạy job khi cần. **Tắt pod ngay sau khi xong.**

### 2.2 Hợp đồng dữ liệu (schema) – KHÔNG tự đổi
Mọi thay đổi schema phải qua NT và tăng số phiên bản (`schema_version`).

**`crawl/status.duckdb` – bảng `crawl_status`**
| Cột | Kiểu | Ghi chú |
|---|---|---|
| `id` | int64 | khóa chính, đúng `id` của BTC |
| `url` | string | URL gốc của BTC (không sửa) |
| `final_url` | string | URL sau redirect |
| `lang_expected` | string | `vi`/`en`/`zh` nếu suy ra được từ file BTC, nếu không thì null |
| `status` | string | `ok` / `http_error` / `timeout` / `dns_error` / `blocked` / `soft_404` / `non_html` / `empty` / `pending` |
| `http_code` | int | |
| `content_type` | string | |
| `n_bytes` | int64 | |
| `attempts` | int | số lần thử |
| `fetched_at` | timestamp | UTC |
| `shard` | int | shard chứa HTML |
| `sha1_raw` | string | hash nội dung thô |

**`crawl/html/shard_XXXXX.jsonl.zst`** – mỗi dòng: `{"id", "final_url", "http_code", "content_type", "encoding", "body_b64"}`. Lưu **bytes gốc** (base64), không decode sai encoding rồi mới lưu.

**`docs/docs_XXXXX.parquet`**
| Cột | Kiểu | Ghi chú |
|---|---|---|
| `id` | int64 | |
| `url`, `final_url` | string | |
| `lang` | string | `vi`/`en`/`zh`/`other`, theo nhận diện thực tế |
| `lang_conf` | float | độ tin cậy nhận diện |
| `title` | string | tiêu đề gốc, không dịch |
| `published_date` | string | ISO `YYYY-MM-DD`, null nếu không có. **Không tự suy.** |
| `site` | string | domain |
| `category` | string | chuyên mục/breadcrumb nếu có, null nếu không |
| `text` | string | văn bản chính, **nguyên văn**, các khối nối bằng `\n\n` |
| `blocks` | list<struct> | danh sách khối theo thứ tự: `{type: heading/paragraph/list_item/table/qa_q/qa_a, level, role, text, char_start, char_end}` với `char_start/char_end` là vị trí trong `text`; `role` ∈ null/`lead`/`caption`/`faq`/`toc`/`reference`/`disclaimer` (xem đặc tả mục 4) |
| `n_chars` | int | |
| `sha1_text` | string | hash của `text` (dùng khử trùng lặp) |
| `dup_group` | int64 | id đại diện nhóm trùng lặp (null nếu không trùng) |
| `extractor` | string | tên + phiên bản bộ trích xuất |
| `quality_flags` | list<string> | ví dụ `short`, `boilerplate_heavy`, `js_required`, `paywall`, `soft_404`, `no_abstract`, `paginated`, `homepage`, `encoding_error` |
| `schema_version` | int | hiện tại = **2** (thêm `role` trong `blocks`) |

**`chunks/<chunk_config>/chunks_XXXXX.parquet`**
| Cột | Kiểu | Ghi chú |
|---|---|---|
| `chunk_id` | string | `"{doc_id}_{chunk_order}"` |
| `doc_id` | int64 | |
| `chunk_order` | int | thứ tự trong tài liệu, từ 0 |
| `text` | string | **phải bằng đúng `doc.text[char_start:char_end]`** |
| `char_start`, `char_end` | int | |
| `heading_path` | string | ví dụ `"Sỏi thận > Triệu chứng"` |
| `chunk_type` | string | `section` / `paragraph` / `qa` / `window` |
| `lang` | string | |
| `n_tokens` | int | theo tokenizer NT chỉ định |
| `chunk_config` | string | tên cấu hình đã sinh ra chunk |

---

## 3. Nhiệm vụ theo giai đoạn

Ký hiệu: **[NT]** nhóm trưởng · **[TV]** thành viên · **DoD** = điều kiện để coi là xong (Definition of Done).

### Giai đoạn 0 – Khởi động (14/10 → 16/10)

#### [TV] T0.1 – Môi trường và tải dữ liệu
**⏰ DL: 23:59 15/10 (T5)**
- Cài Python 3.11, tạo môi trường theo `requirements.txt` của repo.
- Tải `links_corpus_vi_en_zh.parquet` và `query.parquet` từ HF `AIGuruTinix/ViBioMIR` vào `raw/`.
- Thống kê: tổng số URL, số domain, **top 50 domain theo số URL**, phân bố theo ngôn ngữ (nếu suy ra được), tỉ lệ URL PDF hoặc đuôi lạ.
- **DoD:** file `reports/corpus_stats.md` có các thống kê trên, gửi NT.
- **Lưu ý:** không sửa file gốc trong `raw/`. Kiểm tra `id` là số nguyên và không trùng.

#### [TV] T0.2 – Crawl thử 10.000 URL
**⏰ DL: 23:59 16/10 (T6)** – báo cáo pilot là cơ sở để NT chốt kế hoạch crawl ngày 17/10
- Lấy mẫu ngẫu nhiên **phân tầng** theo ngôn ngữ và theo top domain (không lấy 10.000 URL đầu tiên, vì dễ dồn vào một domain).
- Ghi theo schema `crawl_status` + shard HTML.
- **DoD:** báo cáo `reports/pilot_crawl.md` gồm: tỉ lệ `ok` theo ngôn ngữ và theo top domain, phân bố lỗi, tốc độ (URL/phút), dung lượng trung bình mỗi trang, danh sách domain hay lỗi hoặc chặn, ước tính thời gian crawl toàn bộ.
- **Lưu ý:** đây là bước quyết định cả kế hoạch. Báo NT ngay nếu tỉ lệ thành công thấp ở một ngôn ngữ (đặc biệt tiếng Trung).

#### [TV] T0.3 – Chạy các bộ trích xuất trên bộ chuẩn của NT
**⏰ DL: 23:59 16/10 (T6)** – NT chấm và chọn bộ trích trước trưa 17/10
- NT đã dựng xong **bộ chuẩn (gold set) 42 trang** (xem mục 8). TV **không cần tự chấm**, chỉ cần:
  - Chạy ít nhất 2 bộ trích (**trafilatura** chế độ giữ cấu trúc, và **resiliparse**; nếu kịp thì thêm một bộ `selectolax`/`BeautifulSoup` đơn giản) trên đúng 42 trang đó, từ **HTML thô** NT gửi trong file `gold_html.zip` (`html/<id>.html` + `html/<id>.meta.json` + `manifest.csv`; mã hóa ký tự xem `content_type` trong `.meta.json` hoặc thẻ `<meta charset>`). File nén **không kèm đáp án**, để bộ chuẩn luôn là phép thử độc lập; đừng viết luật dựa riêng trên 42 trang này. Riêng 8 trang PubMed thì chạy đường API (đặc tả mục 5.1) thay cho bộ trích HTML.
  - Xuất kết quả theo định dạng `docs` (có `blocks`) vào `reports/extract_gold/<tên_bộ_trích>/<id>.json`.
- **DoD:** đủ file kết quả cho mọi trang và mọi bộ trích; ghi thời gian chạy trung bình mỗi trang của từng bộ.
- **Lưu ý:** không chỉnh tay kết quả. NT chấm bằng script tự động (recall nội dung, tỉ lệ rác).

#### [NT] (để TV biết phụ thuộc)
**⏰ DL của NT:** repo + schema + quyền truy cập: 15/10 · gửi câu hỏi BTC: 15/10 · bộ chuẩn 42 trang: ✅ xong 14/10 · đặc tả làm sạch v1 + danh sách domain: ✅ xong 14/10 (v1.4) · bộ chấm offline: ✅ xong 14/10
- Tạo repo, schema, tài khoản HF/RunPod; hỏi BTC các câu hỏi mở (cách tokenize khi chấm, kiểu `doc_id`, mốc thời gian, domain bị chặn).
- ✅ **Bộ chuẩn 42 trang** + script chấm bộ trích (`eval/score_extraction.py`).
- Viết bộ chấm offline giống luật BTC; khung dev set.
- Gửi TV **đặc tả làm sạch v1** và **danh sách domain ưu tiên** cần luật riêng: **17/10**.

---

### Giai đoạn 1 – Crawl và trích xuất toàn bộ (16/10 → 22/10)

#### [TV] T1.1 – Crawler bản chính
**⏰ DL:** crawler chạy ổn định trên toàn bộ danh sách: **18/10** · ≥ 95% URL đã được thử: **22/10**
Yêu cầu kỹ thuật:
- **Bất đồng bộ** (ví dụ `httpx`/`aiohttp`), có **giới hạn số kết nối đồng thời theo từng domain** (khởi điểm 2–4/domain) và tổng (khởi điểm 100–200; điều chỉnh theo máy và mạng).
- **Timeout** kết nối và đọc (ví dụ 10s/30s), **retry** có backoff (tối đa 3 lần) cho lỗi tạm thời (timeout, 429, 5xx). Không retry 404/410.
- Theo **redirect**, lưu `final_url`. Khóa luôn là `id` gốc.
- User-Agent rõ ràng, nhất quán. Gặp 429 thì tự giảm tốc domain đó.
- **Chạy tiếp được** sau khi dừng: dựa vào `crawl_status`, chỉ lấy URL `pending` hoặc lỗi tạm thời.
- **PubMed (240.000 URL) không crawl HTML** (trang trả 203 đòi cookie/JS): lấy qua **Europe PMC REST API** theo PMID, mỗi request tới 1.000 PMID. NCBI E-utilities đang chặn IP mạng nhà NT nên chỉ là dự phòng (chạy từ máy khác IP). Định dạng văn bản và cách đối chiếu theo đặc tả mục 5.1. **DL lấy xong toàn bộ PubMed: 19/10.**
- **Domain chặn truy cập** (đặc tả mục 5.7): laodong cần đặt cookie `D1N`; vov.vn cần trình duyệt; **familydoctor.com.cn chỉ vào được khi bật Cloudflare WARP (1.1.1.1)** hoặc từ máy cloud; **120ask.com (98.160 URL) chỉ vào được từ IP Nhật/Mỹ/châu Âu** → chạy riêng trên máy cloud (NT cấp); **zysjonline.com: NT đã làm xong từ bản dữ liệu chính thức của site** → file `docs/docs_zysjonline.parquet` (NT gửi), TV **bỏ domain này khỏi danh sách crawl** và gộp file đó khi đẩy `docs/` lên HF. Kiểm tra lại trong T0.2 và báo NT; không tự dùng dịch vụ vượt chặn.
- **Thứ tự ưu tiên** cho phần crawl HTML: theo wave trong đặc tả mục 6 (top 20 domain trước), VI và ZH xen kẽ theo số URL.
- Lưu **bytes gốc**; ghi nhận `content_type` và encoding từ header/meta.
- Nội dung không phải HTML (PDF…): **lưu lại, đánh dấu `non_html`**, chưa xử lý; NT quyết định sau.
- Trang cần JavaScript mới hiện nội dung: đánh dấu `js_required` ở bước trích; **chưa dùng trình duyệt headless** trừ khi NT đồng ý (rất tốn thời gian).
- **DoD:** ≥ 95% URL đã được thử ít nhất một lần trước 22/10; báo cáo hằng ngày (mục 4.2).
- **Lưu ý:**
  - Tiếng Trung có thể chậm hoặc chặn IP nước ngoài, nhưng thường ổn với IP Việt Nam; nếu không ổn, báo NT.
  - Không mở quá nhiều kết nối tới một domain lớn (dễ bị chặn cả domain).
  - Theo dõi dung lượng ổ đĩa; báo NT nếu dự kiến vượt 150GB.

#### [TV] T1.2 – Khung trích xuất và luật theo domain (theo `DAC_TA_LAM_SACH_v1.md`)
**⏰ DL:** khung trích chạy trên mẫu pilot: **18/10** · luật cho nhóm domain ưu tiên đợt 1 + gửi mẫu duyệt: **19/10** · giao mẫu 10K doc cho NT: **19/10** · trích xong toàn bộ trang đã crawl (sau khi NT duyệt): **23/10**

**Quy trình duyệt (bắt buộc, xem chi tiết mục 8):**
1. TV viết luật cho một domain (hoặc nhóm domain) → chạy trên **20 trang mẫu ngẫu nhiên** của domain đó → gửi NT file mẫu + điểm trên bộ chuẩn (nếu domain có trong bộ chuẩn).
2. NT **duyệt hoặc trả lại** kèm ghi chú. **Chỉ domain đã duyệt mới được chạy hàng loạt.**
3. Domain chưa có luật riêng dùng bộ trích chung mà NT đã chọn ở T0.3.
4. Mỗi lần sửa luật: tăng phiên bản `extractor`, chạy lại **bộ chuẩn** để chắc không làm tệ đi các trang khác (kiểm tra hồi quy).

**Quy tắc bắt buộc (từ mentor BTC, KHÔNG được vi phạm):**
1. Bỏ phần giao diện: menu, header/footer, "Trang chủ / Tìm kiếm / Home / Search / 首页 / 搜索 / 相关文章", quảng cáo, bài liên quan, nút chia sẻ, bình luận.
2. **Giữ cấu trúc:** tiêu đề, các mục (heading cấp nào ghi cấp đó), đoạn văn, danh sách, bảng (bảng chuyển thành văn bản theo dòng, giữ thứ tự ô).
3. **Không viết lại, không tóm tắt, không dịch, không sửa chính tả.**
4. **Không bỏ từ phủ định**, không xóa dấu tiếng Việt, **không bỏ stopword**, **giữ nguyên số và đơn vị** (5 mg ≠ 50 mg).
5. Chỉ chuẩn hóa khoảng trắng thừa và giải mã HTML entity. **Không chuẩn hóa Unicode khác** (không NFKC, không đổi dấu câu, không đổi chữ hoa/thường) trong cột `text`.
6. Metadata: tiêu đề gốc, ngày đăng chuẩn ISO (`15/09/2026`, `September 15, 2026`, `2026年9月15日` → `2026-09-15`), chuyên mục nếu có. **Thiếu thì để null, không tự suy** tác giả hay ngày.
7. Nhận diện ngôn ngữ trên `text` (ví dụ fastText `lid.176`), ghi cả `lang_conf`.
8. Trang hỏi đáp (dạng hỏi–đáp nối tiếp): đánh dấu block `qa_q` / `qa_a`.
- **DoD:**
  - Toàn bộ trang `ok` được trích sang `docs/` đúng schema.
  - Kiểm tra tự động: `text` không rỗng, `char_start/char_end` của mọi block khớp với `text`.
  - Báo cáo chất lượng tự động `reports/quality_dashboard.md` (các chỉ số ở mục 8.3), cập nhật mỗi lần chạy.
  - Xuất **mẫu kiểm tra 60 trang ngẫu nhiên** (20 mỗi ngôn ngữ, phân tầng theo domain) dạng HTML so sánh hai cột "trang gốc ↔ văn bản trích" vào `reports/audit/` để NT duyệt bằng mắt.
- **Lưu ý:** ưu tiên viết luật cho các domain **NT đưa vào danh sách** (thường là domain chiếm nhiều URL nhất). Gặp domain lạ chiếm nhiều URL mà chưa có trong danh sách → báo NT.
- **Nguyên tắc khi phân vân:** thà **giữ thừa** một đoạn còn hơn **bỏ sót** nội dung (mất nội dung là mất điểm không lấy lại được).

#### [TV] T1.3 – Kiểm tra chất lượng và khử trùng lặp
**⏰ DL: 23:59 23/10 (T6)**
- Gắn `quality_flags`: `short` (ví dụ < 200 ký tự, NT sẽ chốt ngưỡng), `boilerplate_heavy`, `js_required`, `paywall`, trang lỗi giả (trang trả 200 nhưng nội dung là "Không tìm thấy trang").
- **Trùng lặp chính xác:** cùng `sha1_text` → cùng `dup_group`.
- **Gần trùng:** MinHash/LSH trên `text` (ngưỡng do NT chốt; khởi điểm Jaccard ≥ 0,9). Chỉ **đánh dấu**, **không xóa** bản nào.
- **Quan trọng:** khử trùng lặp **không được làm mất `id`**. Mọi `id` đều phải còn trong `docs/`, chỉ khác là được gắn `dup_group`. (Nhãn chấm điểm dựa trên `id`, mất `id` là mất điểm.)
- Cùng chủ đề ≠ trùng lặp. Bài "Sỏi thận" của hai trang khác nhau, nội dung khác nhau, thì **không** coi là trùng.
- **DoD:** `reports/quality_dedup.md`: số doc theo từng flag, số nhóm trùng, ví dụ 10 nhóm gần trùng để NT kiểm.

#### [NT] (song song)
**⏰ DL của NT:** dev set v0: 20/10 · retrieval trên mẫu 10K: 21/10

- Dev set tổng hợp, bộ chấm offline, các thử nghiệm retrieval trên mẫu 10K doc TV giao.

---

### Giai đoạn 2 – Chunk, công cụ ngôn ngữ, baseline (20/10 → 25/10)

#### [NT] T2.1 – Chunking (NT tự làm)
**⏰ DL của NT:** cấu hình đầu tiên trên mẫu 10K: **20/10** · các cấu hình chính trên toàn bộ doc: **24/10**
- NT thiết kế và cài đặt chunking. TV **không cần làm**, nhưng cần đảm bảo `docs` có `blocks` (loại khối, cấp heading, vị trí ký tự) **chính xác**, vì chunking dựa hoàn toàn vào đó.
- Bất biến NT tự kiểm: `chunk.text == doc.text[char_start:char_end]` trên 100% chunk.
- **TV hỗ trợ khi được yêu cầu:** chạy script chunk của NT trên toàn bộ dữ liệu (trên máy hoặc RunPod), đẩy kết quả lên kho chung.

#### [TV] T2.2 – Công cụ tách từ (thư viện dùng chung)
**⏰ DL: 23:59 21/10 (T4)**
- Viết module `textproc/` gồm:
  - Tách từ tiếng Việt (ví dụ `pyvi` hoặc `underthesea`; benchmark tốc độ trên 10.000 đoạn, báo NT).
  - Tách từ tiếng Trung (ví dụ `jieba`).
  - Tiếng Anh: tokenizer đơn giản + lowercase.
  - Hàm `normalize_for_eval()` theo **đúng luật BTC**: NFKC → giải mã HTML entity → lowercase → gộp khoảng trắng → trim. (Hàm này **chỉ dùng để chấm offline**, không dùng để sửa dữ liệu lưu trữ.)
- **DoD:** có unit test; tài liệu cách dùng trong `textproc/README.md`; tốc độ xử lý ghi trong báo cáo.
- **Lưu ý:** NT quyết định dùng công cụ này ở đâu và như thế nào trong hệ thống.

#### [TV] T2.3 – Đẩy dữ liệu lên kho chung
**⏰ DL:** bản mẫu 10K: **20/10** · bản toàn bộ: **24/10**
- Đẩy `docs/` lên dataset private trên HF (`chunks/` do NT đẩy, hoặc TV đẩy khi NT nhờ). Ghi phiên bản và ngày vào `docs/data_versions.md`.
- **DoD:** NT kéo về được trên RunPod và đọc được bằng `pyarrow`.

#### [NT] (song song)
**⏰ DL của NT:** embedding trên RunPod: 24/10 · **nộp bài đầu tiên: 25/10**

- Baseline retrieval, chạy embedding trên RunPod, **nộp bài đầu tiên**, phân tích lỗi.

---

### Giai đoạn 3 – Tối ưu (24/10 → 31/10)

Ở giai đoạn này **NT trực tiếp làm phần chiến lược** (các kỹ thuật tối ưu điểm). TV hỗ trợ bằng các việc có đặc tả rõ ràng dưới đây.

#### [TV] T3.1 – Hoàn tất crawl còn thiếu
**⏰ DL: 23:59 28/10 (T4)**
- Crawl lại các URL lỗi tạm thời; thử thêm biện pháp NT duyệt cho các domain lớn bị lỗi (ví dụ giảm tốc, đổi giờ crawl).
- **DoD:** báo cáo độ phủ cuối cùng theo ngôn ngữ; danh sách URL không thể lấy được và lý do.

#### [TV] T3.2 – Thu thập nguyên liệu dữ liệu ngoài (chỉ thu thập, NT quyết định cách dùng)
**✅ NT đã làm xong ngày 14/10, TV không cần làm.** Kết quả ở `D:\r2ai_data\external\icd10\icd10_trilingual.parquet` (ICD-10 ba ngôn ngữ) và `D:\r2ai_data\external\terms\wikidata_terms.parquet` (thuật ngữ y khoa vi/en/zh); script trong `terms/`, nguồn trong `docs/data_sources.md`. Yêu cầu ban đầu giữ lại dưới đây để tham chiếu.
**⏰ DL ban đầu: 23:59 26/10 (T2)**
- Bảng mã **ICD-10** (tên bệnh) bản **tiếng Việt, tiếng Anh, tiếng Trung**, kèm mã và nhóm chương. Ghi nguồn và license.
- Danh sách từ đồng nghĩa phổ biến (ví dụ "tiểu đường" / "đái tháo đường") nếu tìm được nguồn mở.
- **DoD:** file sạch dạng bảng (CSV/Parquet: `code, chapter, name_vi, name_en, name_zh, source`) + ghi nguồn trong `docs/data_sources.md`.

#### [TV] T3.3 – Chạy job theo đặc tả trên RunPod
**⏰ DL:** theo từng lần giao; mặc định **trong 24 giờ** kể từ khi NT giao
- Khi NT giao: chạy script đã có sẵn (embedding, tiền xử lý quy mô lớn…) với cấu hình NT chỉ định, ghi log, kiểm tra kết quả, **tắt pod khi xong**.
- **DoD:** log đầy đủ (thời gian chạy, chi phí ước tính, cấu hình), kết quả nằm đúng thư mục.

#### [TV] T3.4 – Gán nhãn thủ công hỗ trợ đánh giá
**⏰ DL: 23:59 29/10 (T5)** (NT gửi hướng dẫn và dữ liệu trước 26/10)
- NT đưa danh sách truy vấn mẫu và các ứng viên hệ thống tìm ra. TV gán nhãn **liên quan / không liên quan** theo **hướng dẫn gán nhãn của NT** (sẽ gửi kèm).
- **Lưu ý:** gán đúng theo hướng dẫn, chỗ nào phân vân thì đánh dấu "không chắc" thay vì đoán. Ví dụ: hỏi "triệu chứng" mà đoạn nói "chẩn đoán" thì **không** liên quan, dù cùng bệnh.
- **DoD:** file nhãn đúng định dạng NT yêu cầu.

#### [NT] (để TV biết)
**⏰ DL của NT:** bài nộp public cuối cùng: **31/10, chậm nhất 20:00** (chừa 4 giờ dự phòng trước hạn 23:59)
- Reranker, cách chọn số lượng kết quả, xử lý truy vấn, trộn tín hiệu, ablation, chọn bài nộp. Chi tiết nằm trong tài liệu chiến lược riêng của NT.

---

### Giai đoạn 4 – Vòng 2 và Demo (01/11 → 11/11)

#### [TV] T4.1 – Đóng gói tái lập
**⏰ DL: 23:59 06/11 (T6)**
- Dockerfile (hoặc môi trường cố định) chạy được toàn bộ pipeline dữ liệu từ `raw/` đến `chunks/`.
- Script một lệnh cho mỗi bước; README hướng dẫn chạy lại từ đầu.
- **DoD:** NT chạy lại được trên một pod sạch mà không cần hỏi thêm.

#### [TV] T4.2 – Phần dữ liệu trong working notes
**⏰ DL: 23:59 05/11 (T5)**
- Viết mục: nguồn dữ liệu, quy trình crawl, trích xuất, làm sạch, khử trùng lặp, thống kê (độ phủ, số doc, số chunk theo ngôn ngữ), các vấn đề gặp và cách xử lý.
- **DoD:** bản nháp gửi NT trước 05/11.

#### [TV] T4.3 – Slide và demo
**⏰ DL: 23:59 09/11 (T2)** – tổng duyệt demo cùng NT ngày 10/11
- Slide phần dữ liệu + sơ đồ pipeline; hỗ trợ NT chuẩn bị demo.

#### [NT]
**⏰ DL của NT:** theo lịch BTC (hạn hệ thống 04/11, đang xác nhận) · working notes hoàn chỉnh: 08/11

- Chốt và nộp các lượt private (chỉ 5 lượt), viết phần phương pháp và kết quả, hoàn thiện working notes, trình bày Demo Day.

---

## 4. Cách làm việc

### 4.1 Git
- Nhánh `main` luôn chạy được. Mỗi việc một nhánh `tv/<tên-việc>` hoặc `nt/<tên-việc>`, merge qua pull request, NT duyệt.
- Commit nhỏ, mô tả rõ. Không commit dữ liệu, không commit secrets.

### 4.2 Báo cáo hằng ngày (TV gửi trước 22:00, ngắn gọn)
```
Ngày:
Đã xong:
Đang làm:
Số liệu (crawl: đã thử / ok / lỗi theo ngôn ngữ; dung lượng):
Vướng mắc (cần NT quyết định):
Đề xuất (ý tưởng, chưa tự làm):
Kế hoạch ngày mai:
```

### 4.3 Họp
- **15 phút mỗi ngày** (giờ cố định do hai người thống nhất).
- Họp dài hơn sau mỗi giai đoạn và sau Mentor Day 3 (22/10).

### 4.4 Nhật ký thí nghiệm
- NT giữ file chung `experiments.md`: mỗi lần chạy/nộp ghi cấu hình, phiên bản dữ liệu (`docs` + `chunk_config`), điểm offline, điểm leaderboard.
- TV ghi phiên bản dữ liệu mỗi lần đẩy lên HF để NT tham chiếu đúng.

### 4.5 Khi gặp sự cố
| Sự cố | Làm gì |
|---|---|
| Domain lớn chặn IP | Dừng domain đó, báo NT, không cố vượt chặn |
| Ổ đĩa sắp đầy | Dừng crawl, báo NT |
| Phát hiện lỗi trích xuất có hệ thống | Ghi ví dụ, báo NT; nếu sửa thì trích lại từ HTML thô (vì vậy phải giữ HTML thô) |
| Pod RunPod treo/tốn tiền bất thường | Tắt pod ngay, báo NT |
| Không chắc một quy tắc | Hỏi NT, đừng đoán |

---

## 5. Lịch tóm tắt

| Ngày | [TV] | [NT] |
|---|---|---|
| 14–16/10 | T0.1 thống kê, T0.2 crawl thử 10K, T0.3 chạy bộ trích trên bộ chuẩn | repo, schema, ✅ **bộ chuẩn 42 trang + script chấm**, hỏi BTC |
| 16–22/10 | T1.1 crawl toàn bộ, T1.2 khung trích + luật domain (chờ NT duyệt), T1.3 chất lượng + trùng lặp | **đặc tả làm sạch v1, duyệt từng domain**, dev set, retrieval trên mẫu |
| 20–25/10 | T2.2 tách từ, T2.3 đẩy HF, hỗ trợ chạy job | **T2.1 chunking**, baseline, embedding RunPod, **nộp bài đầu tiên (25/10)** |
| 22/10 | dự Mentor Day 3 | dự Mentor Day 3, cập nhật kế hoạch |
| 24–31/10 | T3.1 crawl bổ sung, T3.2 ICD-10, T3.3 chạy job, T3.4 gán nhãn | tối ưu, ablation, **nộp public cuối trước 31/10 23:59** |
| 01–11/11 | T4.1 đóng gói, T4.2 viết phần dữ liệu, T4.3 slide | nộp private, working notes, demo |

---

## 6. Checklist bàn giao (TV tự kiểm tra trước khi báo "xong")

- [ ] Đúng schema, đúng kiểu dữ liệu, `id` là int64 và không mất `id` nào.
- [ ] `text` là nguyên văn, không chuẩn hóa ngoài khoảng trắng và HTML entity.
- [ ] Bất biến block: `doc.text[char_start:char_end] == block.text` đúng 100%.
- [ ] Domain đã được NT duyệt trước khi chạy hàng loạt; đã chạy lại bộ chuẩn sau mỗi lần sửa luật.
- [ ] Có báo cáo thống kê và mẫu kiểm tra bằng mắt.
- [ ] Mọi tham số nằm trong config, không hard-code.
- [ ] Đã ghi phiên bản dữ liệu và nguồn dữ liệu ngoài.
- [ ] Không có secrets trong repo; dữ liệu không public.
- [ ] Pod RunPod đã tắt.

---

## 7. Bài học từ những người đi trước (phần dữ liệu)

Tổng hợp từ các nhóm làm dữ liệu web quy mô lớn (Hugging Face FineWeb, TII RefinedWeb), các benchmark trích xuất nội dung, và tài liệu thiết kế crawler. **Số liệu là để tham khảo, không phải luật**: mọi ngưỡng cụ thể vẫn do NT chốt sau khi xem dữ liệu thật.

### 7.1 Trích xuất nội dung
- **Hugging Face (FineWeb) và TII (RefinedWeb)** đều trích từ **HTML thô** (WARC) bằng **trafilatura**, và kết luận rằng trích từ HTML gốc cho chất lượng tốt hơn rõ rệt so với dùng văn bản đã bóc sẵn. → Lý do mình **bắt buộc giữ HTML thô**: trích sai thì trích lại, không cần crawl lại.
- Các benchmark độc lập (Webis 2023, WCXB 2026) cho thấy **trafilatura và resiliparse** là hai bộ trích tốt và ổn định nhất; trafilatura thiên về **sạch** (ít rác, đôi khi mất nội dung), resiliparse thiên về **đủ** (lấy nhiều hơn nhưng lẫn rác). readability-lxml thường kém hơn và đôi khi bỏ sót nội dung rõ ràng. → T0.3 nên so **trafilatura vs resiliparse** là chính.
- Với bài toán này, **mất nội dung tệ hơn lẫn chút rác** (mất đoạn đúng là mất điểm chunk, còn rác thì tầng retrieval lọc được). Khi so sánh ở T0.3, ghi rõ trường hợp **mất nội dung** riêng với trường hợp **lẫn rác**.
- Có nghiên cứu mới (MinerU-HTML, 2025) dùng mô hình để parse HTML và vượt trafilatura, nhưng tốn tài nguyên; chỉ cân nhắc nếu NT yêu cầu.

### 7.2 Crawler
- **Hàng đợi theo từng host**, không dùng một hàng đợi chung (hàng đợi chung dễ dồn dập một trang trong khi bỏ quên trang khác). Mỗi host có bộ đếm thời gian riêng; tốc độ lịch sự thường được nêu khoảng **1–5 request/giây/host** (quy ước thực hành, không phải chuẩn).
- Gặp **429** thì **backoff lũy thừa**; tôn trọng `Crawl-delay` trong robots.txt nếu có; cache robots.txt theo domain (ví dụ 24 giờ).
- **Tách phần tải (IO) và phần parse (CPU)**: tải về lưu HTML thô trước, parse/trích chạy riêng sau. Hai phần hỏng theo cách khác nhau, tách ra thì dễ chạy lại.
- Nếu chạy crawler trên **nhiều máy**, phải **chia theo host** (mỗi host chỉ một máy phụ trách), nếu chia theo URL thì giới hạn tốc độ mỗi host bị nhân lên theo số máy.
- User-Agent rõ ràng, nhất quán.

### 7.3 Khử trùng lặp
- FineWeb dùng **MinHash trên 5-gram** (112 hàm băm) với ngưỡng tương đồng khoảng **0,75**, và thấy khử trùng lặp **trong từng đợt crawl** tốt hơn khử toàn cục cho mục đích huấn luyện LLM. (Ngưỡng lấy từ bản tóm tắt; cần đối chiếu mã `datatrove` trước khi dùng.)
- **Khác biệt quan trọng với mình:** họ khử trùng lặp để **bỏ bớt** dữ liệu; mình khử trùng lặp chỉ để **đánh dấu** (`dup_group`), **tuyệt đối không xóa `id`**, vì BTC chấm theo `id`.

### 7.4 Nhận diện ngôn ngữ
- FineWeb dùng fastText; FineWeb2 so **GlotLID** với **fastText lid.176** và thấy GlotLID có vẻ tốt hơn ở **ngôn ngữ nhiều tài nguyên** (tiếng Việt, tiếng Trung thuộc nhóm này).
- Văn bản **ngắn** (tiêu đề, đoạn ngắn) nhận diện kém hơn nhiều → nhận diện trên **toàn bộ `text`**, không chỉ tiêu đề.
- Chưa có số liệu công bố riêng cho tiếng Việt và tiếng Trung → ở T1.2, **thử cả hai** trên 200 trang đã biết ngôn ngữ, báo NT kết quả.
- Lưu ý trang song ngữ hoặc trộn ngôn ngữ (ví dụ tiếng Việt có chèn thuật ngữ tiếng Anh): ghi `lang_conf` để NT xử lý sau.

---

## 8. Kiểm soát chất lượng làm sạch (NT sở hữu, TV cung cấp số liệu)

Mục tiêu: **NT biết chính xác chất lượng làm sạch bằng con số**, không phải bằng cảm giác, và **không có dữ liệu nào chạy hàng loạt khi chưa được duyệt**.

### 8.1 Bộ chuẩn (gold set) – NT dựng
- **Đã xong 14/10: 42 trang** (18 VI, 16 ZH, 8 PubMed), mỗi top domain một trang, ở `D:\r2ai_data\gold\`. Trang 120ask chưa lấy được nên có trang thay (ask.39.net); familydoctor lấy qua WARP, zysjonline lấy bằng cửa sổ Chrome thật; thêm một trang cnkang và một trang msdmanuals.cn.
- Cách dựng (**không dùng bộ trích nào**, để không chấm một bộ trích bằng chính đầu ra của nó): `eval/gold_tools.py` tải HTML thô (`fetch`), tách mọi khối chữ trên trang và đánh số (`dump`); người duyệt đọc và chọn khoảng khối là nội dung chính trong `selection.json`; `write` chép **nguyên văn** các khối đó ra `gold/<id>.txt`, mỗi khối một đoạn. Đây là "đáp án" của bước làm sạch.
- Trong `<id>.txt`: dòng heading bắt đầu bằng `# `, khối **bắt buộc phải có** bắt đầu bằng `!! ` (1–2 khối mỗi trang, chứa thông tin then chốt như liều, số liệu, câu trả lời của bác sĩ).
- Bộ chuẩn **cố định**, chỉ bổ sung trang mới, không sửa đáp án cũ trừ khi sai.

### 8.2 Script chấm bộ trích – NT viết, TV chạy được
So văn bản trích ra với đáp án, chuẩn hóa giống luật BTC (NFKC, giải mã HTML entity, chữ thường, gộp khoảng trắng), tính trên token:
| Chỉ số | Cách tính | Ý nghĩa | Ngưỡng khởi điểm (NT chốt) |
|---|---|---|---|
| **Recall nội dung** | LCS(trích, đáp án) / độ dài đáp án | giữ được bao nhiêu nội dung đúng | **≥ 0,95** trung bình, không trang nào < 0,85 |
| **Tỉ lệ rác** | 1 − LCS / độ dài văn bản trích | lẫn bao nhiêu thứ không phải nội dung | ≤ 0,15 |
| **Khối bắt buộc** | tỉ lệ khối bắt buộc có trong văn bản trích (≥ 90% token) | không mất mục quan trọng | **100%** |
| **Cấu trúc heading** | heading trong đáp án có xuất hiện như block `heading` không | chunk theo mục cần cái này | ≥ 0,9 |
- Recall quan trọng hơn tỉ lệ rác (mất nội dung là mất điểm vĩnh viễn).
- Script in kết quả theo trang, theo ngôn ngữ, theo domain, và so với phiên bản `extractor` trước (để thấy sửa xong tốt lên hay tệ đi).

### 8.3 Bảng chỉ số chất lượng toàn bộ dữ liệu – TV xuất tự động (`reports/quality_dashboard.md`)
Theo **từng ngôn ngữ** và **top 50 domain**:
- Số trang crawl `ok` / số trang trích ra `text` không rỗng.
- Phân bố độ dài `text` (p5, p50, p95); tỉ lệ trang `short`.
- Tỉ lệ trang có ≥ 1 block `heading`; số block trung bình.
- Tỉ lệ `lang` khác ngôn ngữ dự kiến; `lang_conf` thấp.
- **Dấu hiệu rác:** tần suất các câu xuất hiện ở > 5% trang **cùng domain** (thường là menu, chân trang, quảng cáo còn sót) → liệt kê top 20 câu như vậy cho mỗi domain lớn.
- **Dấu hiệu mất nội dung:** tỉ lệ độ dài `text` / độ dài chữ hiển thị của HTML; trang có tỉ lệ thấp bất thường so với cùng domain.
- **Dấu hiệu lỗi encoding:** tỉ lệ ký tự lạ (ví dụ `�`, chuỗi kiểu `Ã¡`), đặc biệt với trang tiếng Trung.
- Số trang `js_required`, `non_html`, `paywall`, trang lỗi giả.

### 8.4 Duyệt bằng mắt – NT
- Mỗi domain trong danh sách ưu tiên: NT xem **20 trang mẫu** ở dạng hai cột (trang gốc ↔ văn bản trích) do TV xuất ra, đánh dấu lỗi: *mất nội dung*, *lẫn rác*, *sai cấu trúc*, *lỗi encoding*.
- Ngoài ra, mỗi lần chạy toàn bộ: NT xem **60 trang ngẫu nhiên** (20 mỗi ngôn ngữ).
- Kết quả duyệt ghi vào `reports/review_log.md`: domain, phiên bản extractor, quyết định (**duyệt / trả lại**), ghi chú.

### 8.5 Cổng duyệt (quality gate)
1. **Chọn bộ trích chung** (17/10): theo điểm trên bộ chuẩn (mục 8.2).
2. **Mỗi domain ưu tiên**: chỉ chạy hàng loạt khi NT ghi "duyệt" trong `review_log.md`.
3. **Mỗi phiên bản `docs` đẩy lên kho chung**: phải kèm `quality_dashboard.md` và điểm bộ chuẩn **không thấp hơn** phiên bản trước.
4. **Sau Mentor Day 3 / sau mỗi lần nộp bài**: nếu phân tích lỗi cho thấy mất điểm do làm sạch, NT cập nhật đặc tả, TV sửa luật, quay lại bước 2.
