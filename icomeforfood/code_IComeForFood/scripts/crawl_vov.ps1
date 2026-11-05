param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ExtraArgs
)
$ErrorActionPreference='Stop'
$env:PYTHONUTF8='1'
$repoRoot=Split-Path -Parent $PSScriptRoot
$localPython=Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $localPython)) { $localPython='python' }
& $localPython (Join-Path $repoRoot 'crawl\vov.py') @ExtraArgs
exit $LASTEXITCODE
