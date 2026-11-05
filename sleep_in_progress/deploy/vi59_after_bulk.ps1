$env:PYTHONIOENCODING='utf-8'
$env:MODAL_PROFILE='<MODAL_PROFILE_D>'
$o="$env:WORKDIR\r2ai_deliver\cand\vi59d"
$lockPath=Join-Path $o 'finish_controller.lock'
try { $lock=[System.IO.File]::Open($lockPath,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::None) } catch { exit 3 }
try {
  $bulk=Get-Process -Id 23032 -ErrorAction SilentlyContinue
  if ($bulk) { Wait-Process -Id 23032 }
  if (-not (Test-Path (Join-Path $o 'bulk_done.json'))) { exit 4 }
  python deploy/vi59_finish.py
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  python deploy/vi59_final_audit.py
  exit $LASTEXITCODE
} finally { $lock.Dispose(); Remove-Item -LiteralPath $lockPath }
