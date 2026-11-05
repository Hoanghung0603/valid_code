# Hướng dẫn crawl toàn bộ corpus

> Dành cho TV (T0.2 pilot và T1.1 crawler). Viết ngày 14/10/2026 từ số liệu thật: file HF 1.220.598 URL, thử truy cập 43 trang của 33 domain, thử API PubMed.
> Đọc kèm: `PHAN_CONG.md` (schema mục 2.2, nhiệm vụ T0.2/T1.1), `DAC_TA_LAM_SACH_v1.md` mục 5.1 (PubMed) và 5.7 (domain chặn). Bảng kế hoạch theo từng domain: `crawl/domain_plan.csv`.

## 1. Con số cần biết

| Phần | URL | Cách lấy |
|---|---|---|
| PubMed | 240.000 | **API Europe PMC**, không crawl (mục 4) |
| zysjonline.com | 6.591 | **Đã xong** (NT, từ dữ liệu chính thức của site) |
| 120ask.com | 98.160 | Crawl từ **máy cloud ở Mỹ/châu Âu** (NT cấp) |
| Còn lại (91 domain) | 875.847 | Crawl HTML từ máy TV |
| **Tổng phải crawl HTML** | **974.007** | |

- Trang HTML trung bình khoảng **150KB** (đo trên 40 trang thật). Tổng HTML thô khoảng **145GB**; nén zstd còn khoảng 15–25GB.
- Thời gian bị chặn bởi **domain lớn nhất** khi chạy song song mọi domain: suckhoecongdongonline.vn 154.502 URL.

| Domain | URL | Ở 2 req/s | Ở 4 req/s |
|---|---|---|---|
| suckhoecongdongonline.vn | 154.502 | 21,5 giờ | 10,7 giờ |
| 120ask.com (cloud) | 98.160 | 13,6 giờ | 6,8 giờ |
| suckhoedoisong.vn | 85.823 | 11,9 giờ | 6,0 giờ |
| nhathuoclongchau.com.vn | 79.598 | 11,1 giờ | 5,5 giờ |
| thanhnien.vn | 72.111 | 10,0 giờ | 5,0 giờ |

→ Nếu chạy song song tất cả domain thì **khoảng 1 ngày**, cộng lượt thử lại thì **dự trù 2 ngày**. Băng thông trung bình khoảng 2MB/s, mạng nhà là đủ. Muốn kịp DL (≥ 95% URL đã thử trước 22/10) thì **bắt đầu chạy full chậm nhất 18/10**.

## 2. Kiến trúc crawler

```
input (parquet) ─► lập lịch theo domain ─► fetcher bất đồng bộ ─► hàng đợi kết quả ─► writer (1 task)
                   (mỗi domain 1 hàng đợi,   (httpx/aiohttp)                         ├─ shard HTML .jsonl.zst
                    giới hạn req/s riêng)                                              └─ crawl_status (DuckDB)
```

- **Python asyncio + httpx (hoặc aiohttp).** Một tiến trình chạy được 100–200 kết nối cùng lúc là đủ.
- **Mỗi domain một hàng đợi riêng**, có hai giới hạn:
  - số kết nối đồng thời (khởi điểm 4);
  - tốc độ req/s (token bucket; khởi điểm theo cột `start_req_per_s` trong `domain_plan.csv`: 2 với domain ≥ 20.000 URL, 1 với domain nhỏ).
  - Lấy URL **xen kẽ giữa các domain** (round-robin) để mọi domain cùng chạy, không dồn vào một domain.
- **Tự giảm tốc:** gặp 429/503 thì giảm một nửa tốc độ domain đó trong 10 phút. Nếu hơn 20% của 200 request gần nhất bị lỗi thì **tạm dừng domain đó và báo NT** (PHAN_CONG 4.5).
- **Timeout** kết nối 10 giây, đọc 30 giây. **Thử lại** tối đa 3 lần có backoff (5s → 30s → 2 phút) cho timeout, lỗi kết nối, 429, 5xx. **Không thử lại** 404/410.
- **Theo redirect**, lưu `final_url`. Khóa luôn là `id` gốc của BTC.
- **Header giống trình duyệt** (nhiều site chặn User-Agent của Python):
  ```
  User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36
  Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8
  Accept-Language: vi,zh-CN;q=0.9,en;q=0.8
  ```
- **Lưu bytes gốc**, không decode, không làm sạch lúc crawl. Ghi kèm `content_type` để bước trích tự xác định encoding (nhiều trang tiếng Trung dùng gb2312/gbk).
- **Writer chỉ một task duy nhất:**
  - nhận kết quả từ hàng đợi;
  - ghi HTML vào shard hiện tại; đủ 10.000 bản ghi thì sang shard mới;
  - upsert `crawl_status` theo lô (mỗi 1.000 bản ghi hoặc 30 giây).
  - Không cho nhiều task cùng ghi DuckDB.
- **Chạy tiếp được** sau khi dừng: khởi động thì đọc `crawl_status` và bỏ qua id đã `ok` hoặc lỗi vĩnh viễn (404/410). Ghi status **sau khi** HTML đã ghi xong vào shard.
- Ghi log tiến độ mỗi 5 phút theo domain: số đã thử, ok, lỗi theo loại, req/s thực tế. Lấy số này làm báo cáo ngày.

Khung lõi tham khảo (giới hạn tốc độ theo domain):

```python
import asyncio, time, httpx

class DomainLimiter:
    def __init__(self, rate: float, concurrency: int = 4):
        self.interval, self.next_t = 1.0 / rate, 0.0
        self.sem, self.lock = asyncio.Semaphore(concurrency), asyncio.Lock()
    async def __aenter__(self):
        await self.sem.acquire()
        async with self.lock:                      # token bucket: cách đều nhau 1/rate giây
            wait = self.next_t - time.monotonic()
            self.next_t = max(self.next_t, time.monotonic()) + self.interval
        if wait > 0:
            await asyncio.sleep(wait)
    async def __aexit__(self, *exc):
        self.sem.release()

async def fetch(client: httpx.AsyncClient, limiter: DomainLimiter, row) -> dict:
    async with limiter:
        r = await client.get(row.url, follow_redirects=True)
    return {"id": row.id, "final_url": str(r.url), "http_code": r.status_code,
            "content_type": r.headers.get("content-type"), "body": r.content}
```

## 3. Domain cần xử lý riêng (đặc tả mục 5.7, cột `access` trong `domain_plan.csv`)

| `access` | Domain | URL | Làm gì |
|---|---|---|---|
| `ok` | 28 domain đã thử, requests thường lấy được | 697.561 | Crawl bình thường |
| `untested` | 60 domain chưa thử | 110.398 | Pilot kiểm tra; domain nào lỗi thì báo NT |
| `cookie` | laodong.vn | 31.282 | Lần đầu trả HTML khoảng 177 byte có `document.cookie="D1N=<giá trị>"`. Đọc giá trị bằng regex, đặt cookie vào client, gửi lại. Cookie dùng được cho cả phiên |
| `browser` | vov.vn (+ có thể vov2.vov.vn) | 9.784 (+6.899) | Request thường bị 403. Thử trước với **bộ header trình duyệt đầy đủ** (mục 2). Nếu vẫn 403 thì dùng Playwright/Chrome, chạy chậm, một luồng riêng; báo NT trước khi dùng trình duyệt |
| `warp` | familydoctor.com.cn | 26.819 | IP nhà NT bị WAF chặn (403); có thể IP của TV không bị. Nếu 403 thì bật Cloudflare WARP (app 1.1.1.1) **chỉ trong lúc chạy domain này**, hoặc gom vào máy cloud cùng 120ask |
| `cloud` | 120ask.com | 98.160 | Từ Việt Nam và Singapore đều lỗi 521/timeout, WARP cũng không được; từ Nhật/Mỹ/Đức/Hồng Kông thì được. **Chạy cùng crawler trên máy cloud NT cấp** với tham số chỉ domain này, xong chép shard và status về |
| `api` | pubmed.ncbi.nlm.nih.gov | 240.000 | Mục 4 |
| `done` | zysjonline.com | 6.591 | Bỏ khỏi danh sách |

**Không** dùng dịch vụ giải captcha, proxy trả phí hay cách vượt chặn trái điều khoản trang khi chưa có NT duyệt.

## 4. PubMed qua Europe PMC (đã thử ngày 14/10)

- Endpoint `POST https://www.ebi.ac.uk/europepmc/webservices/rest/searchPOST`, form data:
  - `query = EXT_ID:(PMID1 OR PMID2 OR … ) AND SRC:MED` (**1.000 PMID mỗi request**);
  - `resultType = core`, `format = json`, `pageSize = 1000`.
- Kết quả thử: 1.000/1.000 bài trả về, 997 có tóm tắt, 12,5 giây, 6MB JSON. Cả 240.000 PMID cần **240 request**, chạy tuần tự **chưa tới 1 giờ**. Không cần chạy song song, nghỉ 1 giây giữa các request.
- PMID lấy từ URL `https://pubmed.ncbi.nlm.nih.gov/<PMID>/`.
- Lưu JSON gốc mỗi lô vào `crawl/pubmed/batch_XXXX.json.zst`. Bước trích theo đặc tả 5.1 (tiêu đề bỏ dấu chấm cuối, `Nhãn: nội dung`, dòng `Keywords:`, regex xóa thẻ an toàn).
- PMID không có trong kết quả: ghi `status=empty`, báo NT số lượng (NT sẽ thử NCBI qua WARP).
- GET cũng chạy được nhưng chỉ khoảng 200 PMID mỗi request vì URL dài.

## 5. Trình tự chạy

1. **Pilot T0.2 (10.000 URL, trước 16/10):**
   - lấy mẫu phân tầng theo domain, mỗi domain tối thiểu 30 URL (domain ít hơn thì lấy hết), bỏ PubMed và zysjonline;
   - đo theo domain: tỉ lệ `ok`, phân bố mã lỗi, thời gian phản hồi trung bình, KB/trang;
   - mở thử 5 trang `ok` ngẫu nhiên mỗi domain để chắc chắn không phải trang chặn bot hay "không tìm thấy" nhưng vẫn trả 200;
   - báo cáo `reports/pilot_crawl.md` kèm đề xuất tốc độ cho từng domain.
2. **NT duyệt tốc độ** (17/10). Domain không lỗi có thể tăng lên 3–4 req/s.
3. **Chạy full** chậm nhất 18/10. Song song chạy PubMed (mục 4).
4. **Lượt thử lại** cho lỗi tạm thời sau khi lượt đầu xong.
5. **Báo cáo** độ phủ theo domain và ngôn ngữ, danh sách domain còn lỗi.

## 6. Lưu ở đâu

| Dữ liệu | Ở đâu | Dung lượng ước tính |
|---|---|---|
| Input BTC (`links_corpus_vi_en_zh.parquet`, `query.parquet`) | máy TV: `<data>/raw/` | 30MB |
| HTML thô đã nén (`shard_XXXXX.jsonl.zst`, 10.000 trang/shard) | máy TV: `<data>/crawl/html/` | **khoảng 20–25GB** (không nén khoảng 145GB, **không** lưu dạng không nén) |
| Trạng thái từng URL (`status.duckdb`) | máy TV: `<data>/crawl/` | < 1GB, **sao lưu mỗi ngày** |
| JSON PubMed từ Europe PMC | máy TV: `<data>/crawl/pubmed/` | khoảng 1,4GB thô, nén còn vài trăm MB |
| 120ask (crawl trên máy cloud) | ổ mạng (Network Volume) của pod RunPod → nén → tải về `<data>/crawl/html/` máy TV | khoảng 2–3GB nén |
| Văn bản đã trích (`docs/*.parquet`) | máy TV `<data>/docs/` **và đẩy lên dataset Hugging Face private** của đội | vài GB |

- `<data>` là thư mục **ngoài OneDrive/Google Drive** (ví dụ `D:\r2ai_data\`). Thư mục đồng bộ đám mây sẽ chậm và đầy quota.
- **Không** đưa dữ liệu lên GitHub.
- Chỗ chung của hai người là **dataset Hugging Face private** (NT tạo, mời TV). Chỉ đẩy `docs/` (và sau này `chunks/`), không đẩy HTML thô.
- HTML thô giữ ở máy TV đến hết cuộc thi: cần nó để trích lại khi sửa luật làm sạch. Nếu có ổ ngoài thì sao lưu một bản.
- File của NT gửi sang (ví dụ `docs_zysjonline.parquet`) đặt vào `<data>/docs/` rồi đẩy lên HF cùng các file khác.

## 7. Checklist trước khi chạy full

- [ ] Ổ đĩa trống ≥ 100GB (HTML nén + status + docs + dự phòng). Dữ liệu ở ngoài OneDrive.
- [ ] Tắt chế độ ngủ của máy; nếu dùng laptop thì cắm sạc.
- [ ] Chạy thử dừng giữa chừng rồi chạy lại: không tải lại id đã `ok`, không mất bản ghi.
- [ ] Sao lưu `status.duckdb` mỗi ngày.
- [ ] Log tiến độ mỗi 5 phút.
- [ ] Có bảng đếm theo domain để làm báo cáo ngày.



## 8. Long Châu: chỉ ID lẻ, dữ liệu một phần — cập nhật 21/10

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

