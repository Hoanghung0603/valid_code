$env:PYTHONIOENCODING='utf-8'
$env:MODAL_PROFILE='<MODAL_PROFILE_D>'
$lockPath="$env:WORKDIR\r2ai_deliver\cand\vi59d\launch_controller.lock"
try { $lock=[System.IO.File]::Open($lockPath,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::None) } catch { exit 3 }
$py="$env:HIST_TMP\modalenv\Scripts\python.exe"
$old=Get-Process -Id 28524 -ErrorAction SilentlyContinue
if ($old) { Wait-Process -Id 28524 }
& $py deploy/vi59_control.py upload
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py deploy/vi59_control.py spawn --job ping --function ping
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py deploy/vi59_control.py poll --job ping --wait 360
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py deploy/vi59_control.py forecast
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py deploy/vi59_bulk.py
exit $LASTEXITCODE
