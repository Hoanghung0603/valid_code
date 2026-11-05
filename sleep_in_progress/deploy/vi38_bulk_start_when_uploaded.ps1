$ErrorActionPreference = 'Stop'
$upload = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" | Where-Object { $_.CommandLine -like '*python.exe*deploy/vi38_bulk_prepare.py*' }
if (@($upload).Count -gt 1) { throw 'More than one vi38 upload process' }
if ($upload) { Wait-Process -Id $upload.ProcessId }
if (-not (Test-Path "$env:HIST_TMP\vi38\bulk_upload_verified.json")) { throw 'Upload did not finish: no GPU launch' }
if (-not (Test-Path "$env:HIST_TMP\vi38\bulk_token_audit.json")) { throw 'Missing token audit: no GPU launch' }
python deploy/vi38_job_launch.py bulk
exit $LASTEXITCODE
