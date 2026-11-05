param(
    [Parameter(Position=0)]
    [ValidateSet('install','preflight','prepare','plan','crawl','longchau-odd','pubmed','vov','merge-home','extract','status','stop')]
    [string]$Command='preflight',
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ExtraArgs
)
$ErrorActionPreference='Stop'
$env:PYTHONUTF8='1'
$repoRoot=Split-Path -Parent $PSScriptRoot
$localPython=Join-Path $repoRoot '.venv\Scripts\python.exe'
if ($Command -eq 'install') {
    if (-not (Test-Path -LiteralPath $localPython)) {
        # Reuse installed scientific libraries; isolate new local dependencies.
        & python -m venv --system-site-packages (Join-Path $repoRoot '.venv')
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    & $localPython -m pip install -r (Join-Path $repoRoot 'requirements-local.txt')
    exit $LASTEXITCODE
}
if (-not (Test-Path -LiteralPath $localPython)) { $localPython='python' }
& $localPython (Join-Path $PSScriptRoot 'run_local.py') $Command @ExtraArgs
exit $LASTEXITCODE
