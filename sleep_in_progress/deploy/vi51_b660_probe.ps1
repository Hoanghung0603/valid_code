$ErrorActionPreference = 'Stop'
$task = Get-ScheduledTask -TaskName 'r2ai_vi51_prepare' -ErrorAction SilentlyContinue
$info = Get-ScheduledTaskInfo -TaskName 'r2ai_vi51_prepare' -ErrorAction SilentlyContinue
$processes = @(Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -like '*vi51.py*' } | Select-Object ProcessId,ParentProcessId,CommandLine)
$lock = if (Test-Path 'D:\r2ai\locks\heavy_job.lock') { [string](Get-Content 'D:\r2ai\locks\heavy_job.lock' -Raw) } else { $null }
$log = Get-Item 'D:\r2ai\data\sub\vi51\prepare.log' -ErrorAction SilentlyContinue
$report = Test-Path 'D:\r2ai\data\sub\vi51\prepare_report.json'
@{task_state="$($task.State)";last_result=$info.LastTaskResult;processes=$processes;lock=$lock;log_modified="$($log.LastWriteTime)";log_bytes=$log.Length;report=$report} | ConvertTo-Json -Depth 5
