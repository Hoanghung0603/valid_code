# clean_temp_uc.ps1 — delete nodriver uc_* profile dirs with no live chrome process.
# Safe: only dirs matching uc_* under $env:TEMP, only when NO chrome.exe cmdline
# references the dir name, and only when dir was not modified in the last 120s
# (avoids racing a just-started browser). Never kills processes.
$ErrorActionPreference = 'SilentlyContinue'
$log = 'D:\r2ai\data\crawl\tools\clean_temp_uc.log'
function W($m) { Add-Content -Path $log -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" }
$tmp = $env:TEMP
$freeBefore = [math]::Round((Get-PSDrive C).Free/1GB,2)
$cl = @(Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" | ForEach-Object { [string]$_.CommandLine })
$dirs = @(Get-ChildItem -LiteralPath $tmp -Directory -Filter 'uc_*')
$cut = (Get-Date).AddSeconds(-120)
$deleted = 0; $live = 0; $fresh = 0; $failed = 0
$failedNames = @()
foreach ($d in $dirs) {
  $isLive = $false
  foreach ($c in $cl) {
    if ($c -and $c.IndexOf($d.Name, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) { $isLive = $true; break }
  }
  if ($isLive) { $live++; continue }
  if ($d.LastWriteTime -gt $cut) { $fresh++; continue }
  Remove-Item -LiteralPath $d.FullName -Recurse -Force
  if (Test-Path -LiteralPath $d.FullName) { $failed++; $failedNames += $d.Name } else { $deleted++ }
}
$freeAfter = [math]::Round((Get-PSDrive C).Free/1GB,2)
W "pass dirs=$($dirs.Count) deleted=$deleted live=$live fresh=$fresh failed=$failed c_before=$freeBefore c_after=$freeAfter chrome_procs=$($cl.Count)"
if ($failedNames.Count -gt 0) { W ("failed_dirs=" + ($failedNames -join ',')) }
