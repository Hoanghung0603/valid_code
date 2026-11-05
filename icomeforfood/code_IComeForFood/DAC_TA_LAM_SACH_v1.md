# Đặc tả làm sạch và trích xuất – v1.13

> Chủ sở hữu: **Nhóm trưởng (NT)**. Người triển khai: **Thành viên (TV)**. Ngày: 14/10/2026.
> Mọi thay đổi quy tắc phải qua NT và tăng số phiên bản. Đọc kèm `PHAN_CONG.md` (schema mục 2.2, quy trình duyệt mục 8).

---

## 1. Dữ liệu thực tế (phân tích ngày 14/10 trên file HF)

- File `links_corpus_vi_en_zh.parquet` trên HF hiện có **1.220.598 URL**, `id` là int64, không trùng, nằm trong khoảng 583 → 7.466.161.
- **Chỉ có 94 domain.** Top 20 domain chiếm **85,5%**, top 50 domain chiếm **98%** số URL.
  → **Viết luật riêng cho từng domain lớn là khả thi và là cách tốt nhất**, không cần phụ thuộc vào bộ trích tổng quát.
- Theo ngôn ngữ (suy từ domain): VI 740.596 · EN 240.000 · ZH 239.999, cộng 3 URL trang chủ.
- **Toàn bộ 240.000 URL tiếng Anh là PubMed** (`pubmed.ncbi.nlm.nih.gov/<PMID>/`). README của BTC cũng ghi rõ nguồn tiếng Anh là PubMed.
- **Chênh lệch số liệu cần hỏi BTC:** email BTC ghi 1.194.469 URL (VI 714.469), file HF có 1.220.598 (VI 740.598). Phần chênh **đúng bằng 26.129**, bằng chính số URL của **vinmec.com** (các `id` liên tục 733.124 → 759.252). Chưa rõ vinmec bị thêm vào hay bị bỏ ra. **Quyết định của NT (14/10): dùng bản nhiều dữ liệu hơn, tức file HF 1.220.598 URL, crawl cả vinmec.** Vẫn giữ `id` gốc nên tách được vinmec ra nếu BTC trả lời khác.
- Có 3 URL là **trang chủ** (`medlatec.vn/`, `suckhoecongdongonline.vn/`, `baidianfeng.familydoctor.com.cn/`).
- Nhiều bài trên các báo Việt Nam **không phải nội dung y khoa** (ví dụ tin cách ly, kiến trúc). Vẫn trích bình thường, không tự lọc theo chủ đề.
- Truy vấn là **câu hỏi sức khỏe của người dân** (văn nói, có lỗi chính tả, có mô tả kết quả xét nghiệm), trung vị 18 từ, dài nhất 281 từ.
- Danh sách đầy đủ domain kèm số URL và đợt (wave): `D:\r2ai_data\reports\domain_stats.csv`.

## 2. Nguyên tắc chung (áp dụng mọi domain)

1. **Nguyên văn.** Không viết lại, không tóm tắt, không dịch, không sửa chính tả, không chuẩn hóa Unicode (không NFC/NFKC), không đổi hoa/thường, **giữ nguyên dấu câu**.
   - Lý do: bộ chấm của BTC so theo token sau khi chuẩn hóa, và dấu câu dính vào token (ví dụ `lưng,` khác `lưng`). Đổi dấu câu là mất token khớp.
2. **Giữ thừa còn hơn mất.** Khi phân vân một khối có phải nội dung hay không, **giữ lại** và gắn `role` để NT lọc ở bước chunk (mục 4). Mất nội dung là mất điểm vĩnh viễn; nội dung thừa thì lọc được sau.
3. **Giữ cấu trúc.** Mỗi khối (heading, đoạn, mục danh sách, dòng bảng, câu hỏi, câu trả lời) là một block riêng, đúng thứ tự hiển thị.
4. **Chỉ chuẩn hóa khoảng trắng** bên trong block (gộp nhiều khoảng trắng/xuống dòng thành một dấu cách, bỏ khoảng trắng đầu cuối) và giải mã HTML entity.
5. **Không bịa metadata.** Thiếu ngày, chuyên mục thì để null.
6. **Không lọc theo chủ đề**, không bỏ bài vì "không giống y khoa".

## 3. Ghép văn bản

- `text` = các `block.text` nối bằng `"\n\n"`, đúng thứ tự.
- `char_start/char_end` của block là vị trí trong `text`; bất biến `text[char_start:char_end] == block.text` phải đúng 100%.
- **Bảng:** mỗi **dòng** là một block `table`; các ô trong dòng nối bằng `"\t"` (ký tự trắng, không thêm ký tự nhìn thấy được vào văn bản). Dòng tiêu đề bảng cũng là một block.
- **Danh sách:** mỗi `<li>` là một block `list_item` (danh sách lồng nhau giữ thứ tự, `level` = độ sâu).
- **Tiêu đề bài** lưu ở cột `title`, **và** là block `heading` (level 1) đầu tiên nếu trang hiển thị nó.
- Không chèn thêm chữ nào không có trên trang (không thêm "Câu hỏi:", "Trả lời:", số thứ tự tự đánh…).

## 4. Loại block và `role` (schema_version 2)

Thêm trường **tùy chọn** `role` vào mỗi block (string hoặc null). Đây là thay đổi schema → `schema_version = 2`.

| `type` | Dùng cho |
|---|---|
| `heading` | h1–h6 hoặc tiêu đề mục hiển thị như heading (`level` 1–6) |
| `paragraph` | đoạn văn |
| `list_item` | mục danh sách |
| `table` | một dòng bảng |
| `qa_q` | câu hỏi (trang hỏi đáp) |
| `qa_a` | câu trả lời (trang hỏi đáp), mỗi câu trả lời một hoặc nhiều block |

| `role` | Khi nào | Giữ trong `text`? |
|---|---|---|
| `null` | nội dung chính | có |
| `lead` | đoạn sapo/mở đầu in đậm của báo | có |
| `caption` | chú thích ảnh hiển thị dưới ảnh | có |
| `faq` | khối "Câu hỏi thường gặp" trong bài | có |
| `toc` | mục lục đầu bài | có (NT có thể loại khi chunk) |
| `reference` | danh sách "Tài liệu tham khảo / Nguồn tham khảo" | có (NT có thể loại khi chunk) |
| `disclaimer` | câu miễn trừ trách nhiệm cuối bài, lặp lại ở mọi trang | có (NT có thể loại khi chunk) |

**Loại bỏ hoàn toàn (không đưa vào `text`):** menu, header, footer, breadcrumb (chuyển vào `category`), thanh tìm kiếm, nút chia sẻ, tag, bình luận người đọc, "Bài viết liên quan / Xem thêm / Tin cùng chuyên mục" (kể cả dòng "Xem thêm: <tiêu đề bài khác>" chèn giữa bài), hộp quảng cáo, hộp đặt lịch khám/mua thuốc/gọi hotline, thông tin tác giả/byline (chuyển vào metadata nếu có), widget đăng ký nhận tin, popup cookie, alt text của ảnh (không hiển thị).

## 5. Quy tắc theo nhóm domain

Ghi chú: các nhận định về cấu trúc trang dưới đây là **giả thuyết cần kiểm tra trên HTML thật** ở bước pilot. TV báo lại nếu khác.

### 5.1 PubMed – 240.000 URL (19,7%), một domain, ưu tiên cao nhất
- **Không crawl HTML.** Trang PubMed trả mã 203 kèm trang đòi cookie/JavaScript cho mọi request thường; chỉ trình duyệt thật (Chrome headless) mới xem được.
- **Nguồn chính: Europe PMC REST API** (EBI, có toàn bộ bản ghi PubMed), dùng `search` với `query=EXT_ID:(p1 OR p2 …) AND SRC:MED`, `resultType=core`, `format=json`, `pageSize=1000` và phân trang bằng `cursorMark`. Không cần key. Đã thử ngày 14/10: 8/8 PMID, một request.
- **Nguồn thay thế ngang hàng: NCBI E-utilities (EFetch, XML).** IP mạng nhà NT (VNPT) **bị NCBI chặn** ("temporarily blocked due to possible misuse"), nhưng **bật Cloudflare WARP (1.1.1.1) thì gọi được** (đã thử 14/10). Ưu điểm so với Europe PMC: **giữ đúng thứ tự từ khóa như trên trang**. Nhãn mục trong XML viết hoa (`BACKGROUND`); viết `Nhãn: nội dung` với nhãn nào cũng được vì bộ chấm chuyển về chữ thường. Tốc độ: 3 request/giây không key, 10 request/giây có NCBI API key; mỗi request tối đa 200 PMID khi dùng GET. Không tự gửi thư xin gỡ chặn; báo NT.
- **Định dạng `text` phải giống phần hiển thị trên trang PubMed.** Đã đối chiếu trên 8 trang chuẩn: recall 0,996, khớp 100% các khối bắt buộc.
  1. Tiêu đề (`heading` level 1): **bỏ đúng một dấu chấm cuối** (trang không hiển thị dấu này, còn API thì có).
  2. Tóm tắt có cấu trúc: mỗi phần là **một block `paragraph` dạng `Nhãn: nội dung`**, ví dụ `Background: Certain phthalates…`. Nhãn lấy theo `<h4>` của Europe PMC (hoặc thuộc tính `Label` của NCBI, viết như trang hiển thị). Không tách nhãn thành heading riêng.
  3. Tóm tắt không có cấu trúc: một block `paragraph`.
  4. Nếu có từ khóa: thêm một block cuối `Keywords: a; b; c.` (ngăn cách bằng `; `, kết thúc bằng dấu chấm). Europe PMC có thể trả từ khóa **khác thứ tự** so với trang; NCBI XML giữ đúng thứ tự.
  5. Không đưa vào: nhãn "Abstract", tác giả, nơi công tác, PMID/DOI, dòng copyright, "Erratum in", "Similar articles".
  6. **Bẫy khi xóa thẻ HTML:** tóm tắt chứa ký tự `<` thật (ví dụ `<6 mU/l`). Chỉ xóa thẻ thật bằng regex `</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*)?/?>`, rồi mới giải mã entity. Dùng `<[^>]+>` sẽ nuốt mất cả đoạn văn (đã gặp: recall tụt còn 0,875).
- **Đối chiếu bắt buộc:** chấm theo bộ chuẩn (8 trang PubMed trong `D:\r2ai_data\gold`) bằng `eval/score_extraction.py` trước khi chạy toàn bộ. Nếu muốn mở rộng đối chiếu lên 50 trang thì lấy HTML bằng Chrome headless (`eval/gold_tools.py fetch --browser`), không crawl hàng loạt.
- Bài không có tóm tắt: `text` chỉ có tiêu đề, flag `no_abstract`.
- Metadata: ngày xuất bản (theo XML), tên tạp chí vào `category`.
- [Hỏi BTC] xác nhận đoạn tham chiếu PubMed lấy từ phần nào của trang (tiêu đề, tóm tắt, từ khóa) và có chấp nhận nội dung lấy qua API (Europe PMC/NCBI) không.

### 5.2 Hỏi đáp tiếng Trung – `120ask.com` (98.160), `ask.39.net` (32.772)
- Cấu trúc: tiêu đề câu hỏi → mô tả câu hỏi → một hoặc nhiều câu trả lời của bác sĩ.
- Block: tiêu đề câu hỏi = `heading`; mô tả = `qa_q`; mỗi câu trả lời = `qa_a` (giữ **tất cả** câu trả lời, đúng thứ tự hiển thị).
- Loại bỏ: hồ sơ bác sĩ, nút "hữu ích", câu hỏi liên quan, quảng cáo bệnh viện.
- Trang cũ có thể đã bị xóa hoặc **chuyển hướng về trang chủ** → nếu `final_url` là trang chủ hoặc trang danh sách thì đánh lỗi `soft_404`, không trích.
- Có thể có trang chống bot/captcha → flag `blocked`.
- **[14/10] 120ask.com không truy cập được từ Việt Nam:** mọi URL trả 521 (từ nginx của chính site) hoặc quá thời gian chờ, kể cả khi bật WARP. Từ Nhật, Mỹ, Đức, Hồng Kông thì vào được. Xem mục 5.7.

### 5.3 Bài viết tiếng Trung – `cnkang.com`, `familydoctor.com.cn`, các subdomain `*.39.net`, `youlai.cn`, `zhongyibaodian.net`, `a-hospital.com`, `zysjonline.com`, `msdmanuals.cn`, `jbk.39.net`…
- Thân bài, giữ heading mục. Các trang dạng wiki/sách (`a-hospital.com`, `zysjonline.com/books`) thường dài và nhiều mục: giữ đủ cấu trúc heading.
- **Bài chia nhiều trang** (phân trang): kiểm tra ở pilot; nếu có, chỉ trích **trang ứng với URL BTC đưa**, flag `paginated` (NT quyết định có ghép trang hay không).
- **Encoding:** nhiều trang tiếng Trung dùng GBK/GB2312. Ưu tiên charset trong header → `<meta charset>` → dò tự động; nếu là GB2312/GBK thì **decode bằng `gb18030`** (bao trùm). Kiểm tra tỉ lệ ký tự lỗi (`�`) mỗi domain.
- `test.pmphai.com` (2.989 URL, ca bệnh): có thể cần JavaScript, kiểm tra ở pilot.

### 5.4 Bài sức khỏe của bệnh viện/nhà thuốc Việt Nam – `nhathuoclongchau.com.vn`, `vinmec.com`, `medlatec.vn`, `tiemchunglongchau.com.vn`, `hellobacsi.com`, `youmed.vn`, `tamanhhospital.vn`, `bachmai.gov.vn`…
- Bài có cấu trúc rõ (h2/h3 theo mục: nguyên nhân, triệu chứng, chẩn đoán, điều trị) → **giữ heading chính xác**, đây là cơ sở để chunk theo mục.
- Mục lục đầu bài → `role=toc`. Khối "Câu hỏi thường gặp" → `role=faq`, dùng `qa_q`/`qa_a` nếu tách được. "Tài liệu tham khảo" → `role=reference`.
- Loại bỏ: hộp đặt lịch, hotline, sản phẩm gợi ý, banner, thông tin "Tham vấn y khoa: Bác sĩ…" (chuyển vào metadata nếu cần).
- Câu miễn trừ trách nhiệm lặp lại → `role=disclaimer`.

### 5.5 Báo điện tử Việt Nam – `suckhoecongdongonline.vn`, `suckhoedoisong.vn`, `thanhnien.vn`, `laodong.vn`, `phunusuckhoe.giadinhonline.vn`, `giadinhonline.vn`, các báo địa phương, `vietnamnet.vn`, `vov.vn`, `tuoitre.vn`…
- Sapo (đoạn mở đầu) → `role=lead`. Thân bài, heading phụ (nếu có), chú thích ảnh → `role=caption`.
- Loại bỏ: tác giả, nguồn, thời gian (chuyển vào metadata), "Đọc thêm / Tin liên quan" chèn giữa bài, khung bình luận.
- Bài dạng ảnh/video (ít chữ): trích phần chữ có được, flag `short` nếu ngắn.
- Nhiều bài của các báo này có thể đã bị xóa → kiểm tra `soft_404`.

### 5.6 Trường hợp khác
- 3 URL trang chủ: trích bằng bộ chung, flag `homepage`.
- Domain thuộc đợt 3 (ngoài top 50, khoảng 2% URL): dùng bộ trích chung đã chọn ở T0.3, không viết luật riêng.

### 5.7 Trang chặn truy cập (kiểm tra ngày 14/10 trên 43 trang, từ mạng của NT)

| Domain | URL | Hiện tượng | Cách xử lý |
|---|---|---|---|
| `laodong.vn` | 31.282 | HTTP 200, 177 byte, chỉ có script đặt cookie `D1N=…` rồi tải lại | Đọc giá trị cookie trong script, đặt vào session, gửi lại request (đã làm được, xem `eval/gold_tools.py`) |
| `pubmed.ncbi.nlm.nih.gov` | 240.000 | HTTP 203, trang đòi cookie/JavaScript | Không crawl; dùng API (mục 5.1) |
| `vov.vn` | nhỏ | HTTP 403 với request thường | Chrome headless lấy được |
| `120ask.com` | **98.160** | Từ Việt Nam: 521 (nginx của site) hoặc quá thời gian chờ, kể cả trang chủ và cả khi bật WARP. Đo bằng check-host.net: **Nhật, Mỹ, Đức, Hồng Kông trả 200**; Singapore 521; Ấn Độ timeout | Cần **IP lối ra ở Nhật/Mỹ/châu Âu**: chạy crawler 120ask trên một máy cloud (ví dụ pod CPU RunPod ở Mỹ/châu Âu) hoặc qua VPN có server Nhật/Mỹ. WARP **không** giúp được |
| `familydoctor.com.cn` | **26.819** | 403 WAF Alibaba với **IP nhà NT**, nhưng check-host trả 200 từ mọi nước (kể cả TP.HCM) | **Bật Cloudflare WARP (1.1.1.1) là vào được** (đã thử 14/10, ra đúng bài). Hoặc chạy từ máy cloud |
| `zysjonline.com` | 6.591 | Cloudflare "managed challenge" (Turnstile) chặn mọi chương trình tự động ở **mọi** nước: requests, Chrome headless (kể cả đổi User-Agent), WARP. **Cửa sổ Chrome thật điều khiển qua DevTools thì qua**: trang đầu khoảng 5 giây, các trang sau khoảng 1 giây nhờ cookie. **Site có bản dữ liệu chính thức** ở link "网站数据下载" (`/download/`): `zysj-data.zip` 249MB, 4 bảng SQL (`zysjllsj` nội dung sách, `zysjcell` danh mục sách, `zysjyj` phương thuốc, `zysjzhsj` bài tổng hợp), đã tải về `D:\r2ai_data\external\zysjonline\` | ✅ **Xong 14/10 (NT), không crawl.** `extract/zysjonline_dump.py` ghép **6.591/6.591 URL** với bản dữ liệu và xuất `D:\r2ai_data\docs\docs_zysjonline.parquet` (schema v2, `extractor=zysj_dump_v1`). Đối chiếu với 20 trang thật (đủ loại books/articles/columns + trang chuẩn): nội dung khớp 100% |

- 120ask và familydoctor cộng lại khoảng **125.000 URL, tức khoảng 52% phần tiếng Trung**; đã có đường lấy cho cả hai. TV kiểm tra lại trong T0.2 và báo NT. **Không** dùng dịch vụ vượt captcha trả phí hoặc cách vượt chặn trái điều khoản trang khi chưa có NT duyệt.
- **IP nhà NT (VNPT) đang bị một số nơi gắn cờ** (NCBI, WAF của familydoctor), có thể do IP dùng chung của nhà mạng. Nếu domain nào báo 403 bất thường, thử lại khi bật WARP trước khi kết luận là bị chặn.
- Wayback Machine không lưu các trang đã thử của ba domain này.
- Mỗi URL không lấy được vẫn có một dòng trong `crawl_status` (`status=blocked` hoặc `http_error`) để biết độ phủ.
- **[19/10] Các subdomain bài viết của 39.net** (`care`, `fk`, `heart`, `baby`, `nk`, `cancer`, `tnb`, `woman`, `food`, `ek`, `man`, `shen`, `gan`, `wei`, `sj`, `fitness`, `gk`, `pf`, `gc`, `byby`): khi crawl nhanh, site trả trang xác minh `image.39.net/verify.html` với HTTP 200. Tới 19/10 có **9.310 trang** như vậy bị ghi `ok`. Từ máy NT, các URL này vẫn mở ra bài thật. Cách xử lý: coi là `blocked` (mục 7), crawl lại chậm.

## 6. Thứ tự ưu tiên viết luật riêng (wave)

| Wave | Domain | Phủ | Cách làm |
|---|---|---|---|
| 1 | Top 20 (gồm PubMed qua API) | 85,5% | **Luật riêng từng domain** + NT duyệt 20 trang mẫu trước khi chạy hàng loạt |
| 2 | Hạng 21–50 | +12,5% (cộng dồn 98%) | Bộ trích chung; NT xem 10 trang mẫu mỗi domain; chỉ viết luật riêng nếu trượt ngưỡng |
| 3 | Còn lại (44 domain) | ~2% | Bộ trích chung, kiểm tra qua bảng chỉ số tổng |

Thứ tự trong wave 1 (theo số URL): pubmed (API) → suckhoecongdongonline.vn → 120ask.com → suckhoedoisong.vn → nhathuoclongchau.com.vn → thanhnien.vn → cnkang.com → ask.39.net → laodong.vn → phunusuckhoe.giadinhonline.vn → familydoctor.com.cn → vinmec.com → medlatec.vn → giadinhonline.vn → baohaiphong.vn → baonghean.vn → vietnamnet.vn → baodanang.vn → tiemchunglongchau.com.vn → baocantho.com.vn.

## 7. Phát hiện trang lỗi

| Dấu hiệu | `status` / flag |
|---|---|
| HTTP 200 nhưng nội dung là "không tìm thấy / 404 / 页面不存在 / 该问题已被删除" | `soft_404` |
| vinmec trả **hộp chọn ngôn ngữ** thay cho bài: có câu `Unfortunately, the content on this page is not available in English`, tiêu đề chỉ còn tên site, không có `div#main-article` (slide Mentor 12/10, trang 13) | `soft_404`; tải lại với `Accept-Language: vi` |
| Chuyển hướng về trang chủ hoặc trang danh sách | `soft_404` |
| Trang captcha, xác minh, chặn bot. Ví dụ 39.net: `final_url` là `https://image.39.net/verify.html?referer=…`, trang khoảng 2,3KB, tiêu đề và văn bản chỉ còn `39健康` | `blocked`; crawler ghi `status=blocked` (không phải `ok`) để crawl lại |
| HTML có nhưng thân bài rỗng, nội dung nạp bằng JavaScript | `js_required` |
| Văn bản sau trích < ngưỡng `short` (khởi điểm 200 ký tự, NT chốt) | `short` |
| Tỉ lệ ký tự lỗi encoding cao | `encoding_error` |

Không xóa bản ghi lỗi: mọi `id` vẫn có một dòng trong `docs/` (text rỗng nếu không trích được) để biết rõ độ phủ.

## 8. Metadata

- `published_date`: ưu tiên `meta[property=article:published_time]` → JSON-LD `datePublished` → ngày hiển thị trên trang; chuẩn hóa `YYYY-MM-DD` (`15/09/2026`, `September 15, 2026`, `2026年9月15日` → `2026-09-15`). Không có thì null.
- `category`: breadcrumb hoặc chuyên mục hiển thị; PubMed: tên tạp chí.
- `lang`: chạy cả **GlotLID** và **fastText lid.176** trên 200 trang đã biết ngôn ngữ (theo domain), báo NT kết quả; NT chọn một. Ghi `lang_conf`. Ngôn ngữ dự kiến theo domain lưu ở `lang_expected` (bảng `domain_stats.csv`).

## 9. Kiểm soát chất lượng

Theo `PHAN_CONG.md` mục 8: bộ chuẩn **42 trang** (`D:\r2ai_data\gold\`, **đã xong 14/10**: 18 VI, 16 ZH, 8 PubMed; trang 120ask chưa lấy được nên có trang thay; familydoctor lấy qua WARP; zysjonline lấy bằng cửa sổ Chrome thật), script `eval/score_extraction.py`, bảng chỉ số tổng, duyệt theo domain, cổng duyệt.

- HTML thô của 40 trang nằm ở `D:\r2ai_data\gold\html\<id>.html` (kèm `<id>.meta.json`). **Chạy bộ trích trên chính các file này** để mọi bộ trích được so trên cùng một bản HTML.
- Bộ chuẩn được dựng **không qua bộ trích nào**: `eval/gold_tools.py dump` tách trang thành mọi khối chữ có đánh số, người duyệt chọn khoảng khối ở `selection.json`, rồi `write` chép nguyên văn ra `<id>.txt`. Lý do: nếu dựng bằng trafilatura thì sẽ chấm trafilatura bằng chính đầu ra của nó.
- Kiểm tra nhanh đã chạy: bộ "lấy toàn bộ chữ" có recall 1,000 nhưng tỉ lệ rác 0,593 nên **trượt** cổng duyệt; bộ "chép đúng bộ chuẩn" đạt 1,000 và **qua**.

Lệnh chấm:
```
python eval/score_extraction.py --gold D:/r2ai_data/gold --pred D:/r2ai_data/reports/extract_gold --out D:/r2ai_data/reports/extract_scores
```
Định dạng file trích cho bộ chuẩn: `reports/extract_gold/<tên_bộ_trích>/<id>.json` gồm `{"id", "text", "blocks"}` đúng schema.

## 10. Câu hỏi đang chờ BTC

1. File HF 1.220.598 URL hay email 1.194.469 URL là bản chính thức? (chênh đúng bằng vinmec.com). *Không chặn việc: NT đã chọn bản 1.220.598.*
2. Đoạn tham chiếu của PubMed lấy từ phần nào của trang? Dùng API của NCBI để lấy nội dung có được chấp nhận không?
3. Bộ chấm tách token thế nào (theo khoảng trắng hay theo ký tự với tiếng Trung)?
4. `120ask.com` (98.160 URL) không truy cập được từ Việt Nam (vào được từ Nhật/Mỹ/châu Âu). BTC có bản nội dung các trang này không, hoặc có hướng dẫn gì?

## Lịch sử thay đổi
- v1 (14/10/2026): bản đầu tiên, dựa trên phân tích file HF và hướng dẫn Mentor Day 2.
- v1.1 (14/10/2026): PubMed chuyển sang Europe PMC (NCBI chặn IP), quy định định dạng PubMed giống trang; thêm mục 5.7 về các trang chặn truy cập; bộ chuẩn 40 trang đã xong; thêm câu hỏi BTC số 4.
- v1.2 (14/10/2026): chốt dùng bản 1.220.598 URL; kết quả thử WARP và đo từ nhiều nước (familydoctor và NCBI vào được qua WARP; 120ask cần IP Nhật/Mỹ/châu Âu); bộ chuẩn thêm trang familydoctor (41 trang).
- v1.3 (14/10/2026): zysjonline qua được bằng cửa sổ Chrome thật và có bản dữ liệu chính thức của site; NT tự xử lý domain này; bộ chuẩn thêm trang zysjonline (42 trang).
- v1.4 (14/10/2026): zysjonline xong từ bản dữ liệu chính thức (6.591/6.591 doc, khớp 100% trên 20 trang thật).
- v1.5 (17/10/2026): thêm dấu hiệu `soft_404` cho hộp chọn ngôn ngữ của vinmec; bộ lọc rác dùng chung `extract/postfilter.py` (xem `reviews/2026-10-17_v3_1017.md` mục 3).
- v1.6 (17/10/2026): **khử lặp chuyển hẳn vào `extract/postfilter.py`**; bộ trích không tự khử lặp. Chỉ bỏ cả đoạn lặp (≥ 3 khối liền nhau) hoặc khối dài (≥ 100 ký tự) đã xuất hiện; lặp ngắn (tiêu đề nhắc lại, mục lục và heading, chú thích `Ảnh minh họa.`) là nội dung thật. Bộ lọc thêm luật danh sách tin liên quan cuối bài, hộp sản phẩm, footer Long Châu (xem `reviews/2026-10-17_v3_2116.md`).
- v1.7 (18/10/2026): `postfilter.py` bỏ heading giống hệt heading liền trước (tiêu đề bị xuất 2 lần), bắt dòng nguồn có tên viết tắt (`T. Linh (Theo …)`), và lọc lặp đến khi ổn định (cắt danh sách tin liên quan xong thì dòng tác giả lộ ra ở cuối bài). Xem `reviews/2026-10-18_v3_2335.md`.
- v1.8 (18/10/2026): `postfilter.py` chỉ bỏ nhãn chân trang (`Giám đốc`, `Giấy phép`, `Đường dây nóng`, `Trụ sở`…) ở dạng `Nhãn: giá trị` hoặc đứng một mình, vì "Giám đốc Bệnh viện … cho biết" là nội dung; bắt hộp đặt lịch vinmec dài hơn 200 ký tự. Bộ chuẩn 2: một đoạn văn bị tác giả lặp nguyên văn thì bỏ bản sau không tính là mất nội dung. Xem `reviews/2026-10-18_v3_gold2.md`.
- v1.9 (18/10/2026): `postfilter.py` bỏ thêm hộp tin liên quan chèn trong bài (≥ 2 cặp tiêu đề + sapo bị cắt bằng `...` ở cuối bài), khối quảng bá app và dòng tác giả / chuyên khoa / ngày đăng của youmed, shortcode WordPress chưa render, hộp và nhãn `疾病百科`, thẻ bác sĩ tên dài, dòng thẻ chủ đề của ask.39. Xem `reviews/2026-10-18_v3_gold2.md` mục 9.
- v1.10 (19/10/2026): `crawl/container_hints.csv` thêm hint cho 26 domain chưa có hint, mỗi domain có ≥ 1.000 URL (khoảng 95K URL), và nhận 4 hint trước đây ghi đè trong notebook (jbk.39, msdmanuals, phunusuckhoe, familydoctor). `postfilter.py` giữ dòng `Ngày tham khảo: …` / `Ngày truy cập: …` trong tài liệu tham khảo. Xem `reviews/2026-10-18_v3_1951.md` mục 11.
- v1.11 (19/10/2026): bộ chuẩn 3 (24 trang, `D:/r2ai_data/gold3`). Hints thêm `also` sapo cho qdnd, hint cho baothanhhoa, wujue và 16 domain con 39.net cùng mẫu `div.art_content`. `postfilter.py` bỏ hộp "HỆ THỐNG BỆNH VIỆN ĐA KHOA TÂM ANH" và danh sách địa chỉ sau nó, dòng "Cập nhật lần cuối: …", dòng "… nguồn ưu tiên trên Google …", dòng người nhập bài `文章录入：` của wujue. Xem `reviews/2026-10-18_v3_1951.md` mục 12.
- v1.12 (19/10/2026): duyệt mẫu thử 2 (40K). Hints thêm `120ask.com` (`div.b_askcont` + `div.b_anscont_cont`) và `jb39.com`. `postfilter.py` bỏ nút `投诉`, danh sách `其他类似问题`, dòng `发表于 …` của 120ask; dòng `N条追问`, `回复追问：`, `取消 提交` của ask.39; hộp `…医生推荐` của youlai. Trang xác minh của 39.net là `blocked` (mục 5.7, 7). Xem `reviews/2026-10-19_v3_sample2.md`.
- v1.13 (21/10/2026): `postfilter.py` bỏ ô "Đánh giá độ hữu ích của bài viết" (cùng các nhãn đi liền sau) và banner cookie của nhathuoclongchau, hai thứ chỉ có trong HTML đã chạy JS (crawl4ai). Chỉ bỏ đúng các khối đó, không cắt phần sau, vì FAQ nằm sau ô đánh giá. Crawler phải lưu body gốc của response. Trang `/benh/` của Long Châu: câu hỏi FAQ dạng accordion là `qa_q` (mục 5.4). Xem `reviews/2026-10-21_pubmed_longchau.md`.
