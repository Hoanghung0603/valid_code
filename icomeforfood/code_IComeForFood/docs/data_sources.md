# Nguồn dữ liệu và dịch vụ ngoài

Mỗi nguồn ngoài dùng trong pipeline phải có một dòng ở đây (quy định 2 trong `PHAN_CONG.md`): tên, URL, license/điều khoản, ngày tải, dùng để làm gì.

| Nguồn | URL | License / điều khoản | Ngày tải / thử | Dùng để |
|---|---|---|---|---|
| ViBioMIR – danh sách URL và truy vấn của BTC | https://huggingface.co/datasets/AIGuruTinix/ViBioMIR | CC BY-NC 4.0 (theo README) | 14/10/2026 | Corpus `links_corpus_vi_en_zh.parquet` (1.220.598 URL), `query.parquet` |
| Europe PMC REST API | https://europepmc.org/RestfulWebService | Điều khoản Europe PMC; bản ghi PubMed | thử 14/10/2026 | Tiêu đề, tóm tắt, từ khóa của 240.000 bài PubMed trong corpus |
| Bản dữ liệu chính thức của zysjonline.com (中醫世家) | https://zysjonline.com/download/ (link "网站数据下载" ở chân trang) | Site cho tải công khai; chưa thấy ghi license, cần kiểm tra | tải 14/10/2026 (`zysj-data.zip`, 249MB) | Nội dung các trang zysjonline trong corpus, thay cho crawl (nếu ánh xạ được URL) |
| ICD-10 tiếng Việt (Bộ Y tế, QĐ 4469/QĐ-BYT 2020 và Thông tư 06/2026/TT-BYT) | https://icd.kcb.vn (API công khai mà trang dùng: `ccs.whiteneuron.com/api/ICD10/...`) | Văn bản của Bộ Y tế, tra cứu công khai | 14/10/2026 | Tên bệnh tiếng Việt theo mã ICD-10 (`terms/icd10_vn.py`) |
| ICD-10 WHO 2019 tiếng Anh | PyPI `simple-icd-10` 2.1.1 (dữ liệu ICD-10 2019 của WHO) | © WHO; dùng cho nghiên cứu | 14/10/2026 | Tên bệnh tiếng Anh, cây chương/nhóm |
| ICD-10 tiếng Trung: 医保版 v2.0 (国家医疗保障局), 国标 2020 对照, 中医疾病/证候分类与代码 (GB/T 15657) | Bản sao bảng chính thức trên https://gitee.com/DoctorLi2023/icd-10 | Bảng của cơ quan nhà nước TQ; bản sao không chính thức, cần đối chiếu nếu dùng làm chuẩn | 14/10/2026 | Tên bệnh tiếng Trung, 34.854 tên chẩn đoán chi tiết, mã Đông y (`terms/icd10_zh.py`) |
| Wikidata | https://query.wikidata.org | CC0 | 14/10/2026 | Tên và tên gọi khác vi/en/zh của mục có mã MeSH, ICD-10, ATC (`terms/wikidata_terms.py`) |
| NCBI E-utilities (EFetch) | https://www.ncbi.nlm.nih.gov/books/NBK25497/ | Chính sách sử dụng của NCBI (≤ 3 request/giây, 10 nếu có API key) | thử 14/10/2026 | Nguồn PubMed thay thế, giữ đúng thứ tự từ khóa |

## Mô hình

Luật BTC: chỉ mô hình mở, ≤ 15B tham số, phát hành trước 01/08/2026.

| Mô hình | URL | License | Phát hành / tham số | Dùng để |
|---|---|---|---|---|
| Qwen3.5-9B (bản GGUF Q4_K_M của unsloth, chạy bằng llama.cpp) | https://huggingface.co/Qwen/Qwen3.5-9B | Apache 2.0 | 27/02/2026 / 9,65B | Sinh và kiểm câu hỏi cho dev set tổng hợp (`eval/make_devset.py`) |
| BGE-M3 | https://huggingface.co/BAAI/bge-m3 | MIT | 01/2024 / 568M | Embedding dense cho retrieval (`retrieval/baseline.py`) |
