# Best-effort: remove the 72h default ExecutionTimeLimit from our own task so
# the unattended multi-cycle supervisor is never hard-killed mid-run.
$ErrorActionPreference = 'SilentlyContinue'
$name = 'r2ai_crawl_async_recovery'
try {
  $s = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 0
  Set-ScheduledTask -TaskName $name -Settings $s | Out-Null
  $cur = (Get-ScheduledTask -TaskName $name).Settings.ExecutionTimeLimit
  Write-Output ("execution_time_limit=" + $cur)
} catch {
  Write-Output ("execution_time_limit=unchanged:" + $_.Exception.Message)
}
exit 0
