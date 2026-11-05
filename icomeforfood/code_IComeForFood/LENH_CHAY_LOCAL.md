# Lệnh terminal chạy local — Long Châu chỉ ID lẻ

Muốn dùng Crawl4AI, xem [hướng dẫn Crawl4AI](docs/longchau_crawl4ai.md) và thêm
`--engine crawl4ai` vào lệnh `longchau-odd`. Các lệnh bên dưới mặc định dùng HTTPX.

Dùng **PowerShell** trên máy hiện tại. Chỉ lấy `id % 2 == 1` của `nhathuoclongchau.com.vn`: ID có trong bảng `crawl_status` của DB nguồn, không phải số ở cuối URL. Không cần corpus Parquet. Dữ liệu Long Châu được tính là **một phần**.

Lệnh hiện tại lấy trực tiếp ID/URL từ DB, gồm các ID từng bị chặn khi cooldown đã hết:

```powershell
.\scripts\run_local.ps1 longchau-odd --status-db '.\status.duckdb' --scan --retry-blocked
```

Nếu DB tên `statusduck.db`, chỉ thay tên file ở `--status-db`. Xem `docs/longchau_db_local.md`.


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

## 1. Vào repo và kiểm tra

```powershell
cd "D:\New folder (3)\r2aiIComeForFood26"

.\scripts\run_local.ps1 preflight
.\scripts\run_local.ps1 longchau-odd
```

Lệnh `longchau-odd` không có cờ chỉ lập danh sách offline, **không gửi request**.

Môi trường trên máy này đã cài xong. Nếu cài trên máy mới hoặc thiếu thư viện:

```powershell
.\scripts\run_local.ps1 install
.\scripts\run_local.ps1 prepare
```

## 2. Xem cooldown trước khi thử

```powershell
$cooldownPath = "D:\r2ai_data\crawl\longchau_odd\cooldown.json"

if (Test-Path -LiteralPath $cooldownPath) {
    $cooldownState = Get-Content -LiteralPath $cooldownPath -Raw | ConvertFrom-Json
    $cooldownState
    if ($cooldownState.next_probe_at) {
        "Thời điểm được thử tiếp, giờ Việt Nam:"
        [DateTimeOffset]::Parse($cooldownState.next_probe_at).ToOffset([TimeSpan]::FromHours(7))
    }
}
```

Lượt thử ban đầu ngày 21/10 nhận HTTP 403 sau đúng một request. Đọc file trên để biết thời điểm cooldown hiện tại; không xóa hoặc sửa file để bỏ thời gian nghỉ.

## 3. Thử đúng một request

Chỉ chạy khi đã hết cooldown:

```powershell
.\scripts\run_local.ps1 longchau-odd --probe
```

- Nếu lấy được HTML HTTP 200, kết quả có `"ok": 1` và trạng thái `probe_ok`: chuyển sang bước 4.
- Nếu nhận 403/429/503 hoặc trang xác minh, tiến trình dừng ngay và lưu cooldown ít nhất 6 giờ. Không chạy tiếp bước 4.
- Timeout, redirect hoặc kết quả khác HTTP 200 không được coi là probe thành công.
- Probe không tự chạy tiếp hàng đợi.

## 4. Crawl một lượt nhỏ, chỉ sau khi probe thành công

```powershell
.\scripts\run_local.ps1 longchau-odd --run --batch-size 50 --delay 30
```

Một luồng, tối thiểu 30 giây giữa các request, tối đa 50 URL/lượt. Hết lượt thì nghỉ 3 giờ; hết thời gian nghỉ có thể chạy lại cùng lệnh. ID đã có raw/ok được bỏ qua.

Nếu bị chặn: dừng toàn bộ luồng này, đợi cooldown rồi quay về bước 3. Không reset blocked về pending. Không dùng VPN, proxy, đổi IP hoặc giải CAPTCHA. Tốc độ thấp không bảo đảm tránh được chặn.

Có thể chạy ít trang hơn và chậm hơn:

```powershell
.\scripts\run_local.ps1 longchau-odd --run --batch-size 10 --delay 60
```

Hai lệnh ở bước này là **hai lựa chọn**, chỉ chạy một lệnh; không mở nhiều instance.

## 5. Xem log và kết quả

Mở PowerShell thứ hai:

```powershell
Get-Content -LiteralPath "D:\r2ai_data\crawl\longchau_odd\crawl.log" -Tail 30 -Wait
```

Ctrl+C trong cửa sổ xem log chỉ dừng việc xem log.

Kết quả của lượt gần nhất:

```powershell
Get-Content -LiteralPath "D:\r2ai_data\crawl\longchau_odd\run_summary.json" -Raw
```

Đếm trạng thái mới nhất của mỗi ID, bỏ dòng bị cắt:

```powershell
$latestById = @{}
Import-Csv -LiteralPath "D:\r2ai_data\crawl\longchau_odd\status.csv" | ForEach-Object {
    if ($_.id -match '^\d+$' -and $_.status) {
        $latestById[$_.id] = $_
    }
}
$latestById.Values | Group-Object status | Select-Object Name, Count
```

HTML thô đã nén nằm ở:

```text
D:\r2ai_data\crawl\longchau_odd\html\home_shard_*.jsonl.zst
```

Mỗi record có đúng 6 trường: `id`, `final_url`, `http_code`, `content_type`, `encoding`, `body_b64`. Đây là HTML chưa clean; file status và log nằm cạnh thư mục `html`.

## 6. Dừng mềm và chạy tiếp

Ở terminal thứ hai, chỉ dừng luồng Long Châu này:

```powershell
New-Item -ItemType File -Path "D:\r2ai_data\crawl\longchau_odd\STOP" -Force
```

Đợi terminal crawler lưu xong shard và thoát. Khi muốn chạy lại có chủ ý, xóa riêng STOP:

```powershell
Remove-Item -LiteralPath "D:\r2ai_data\crawl\longchau_odd\STOP" -ErrorAction SilentlyContinue
```

Việc xóa STOP không bỏ cooldown. Kiểm tra lại bước 2; sau khi bị chặn vẫn phải probe một request trước khi chạy batch mới.

## 7. Hợp nhất raw vào pipeline

Chạy sau khi crawler đã dừng, có shard thành công và không có tiến trình khác đang ghi DB:

```powershell
cd "D:\New folder (3)\r2aiIComeForFood26"

.\scripts\run_local.ps1 merge-home --home-dir "D:\r2ai_data\crawl\longchau_odd"
```

Lệnh đổi số shard vào `D:\r2ai_data\crawl\html` và cập nhật DB đang được CLI chọn. Chạy lại không chép trùng shard. Không đưa raw HTML hoặc DB lên GitHub.

## Nếu PowerShell chặn chạy file .ps1

Dùng Python trong môi trường local trực tiếp, không cần đổi execution policy:

```powershell
cd "D:\New folder (3)\r2aiIComeForFood26"

.\.venv\Scripts\python.exe .\scripts\run_local.py longchau-odd
.\.venv\Scripts\python.exe .\scripts\run_local.py longchau-odd --probe
# Chỉ chạy dòng sau nếu probe thành công:
.\.venv\Scripts\python.exe .\scripts\run_local.py longchau-odd --run --batch-size 50 --delay 30
```

## Các luồng khác — chạy riêng khi cần

Các lệnh này không dùng để crawl Long Châu:

```powershell
# PubMed qua API:
.\scripts\run_local.ps1 pubmed

# Các domain còn lại, loại Long Châu:
.\scripts\run_local.ps1 crawl --workers 32

# Xem trạng thái sau khi tiến trình ghi DB đã dừng:
.\scripts\run_local.ps1 status

# Trích full v26 sau khi đã đồng bộ HTML cũ và chuẩn bị Gold:
.\scripts\run_local.ps1 extract
```

Để trích full, cần shard HTML cũ từ Drive/VPS trong `D:\r2ai_data\crawl\html`; DB trạng thái không chứa HTML.

