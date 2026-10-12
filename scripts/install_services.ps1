# Registers NeuroTrade Dashboard + Screener Watch as logon-triggered,
# auto-restarting Windows scheduled tasks.
#
# RUN THIS AS ADMINISTRATOR:
#   1. Press Start, type "PowerShell"
#   2. Right-click "Windows PowerShell" -> "Run as administrator"
#   3. Paste:
#        powershell -ExecutionPolicy Bypass -File "C:\Users\hassa\Downloads\NueroTrade\scripts\install_services.ps1"
#
# Idempotent — safe to re-run (overwrites existing task definitions).

$ErrorActionPreference = "Stop"
$repo = "C:\Users\hassa\Downloads\NueroTrade"
$uid  = "S-1-5-21-3387166403-4277222955-1062815531-1001"

$tasks = @(
  @{ Name = "NeuroTrade Dashboard";      Bat = "$repo\scripts\run_dashboard_task.bat" },
  @{ Name = "NeuroTrade Screener Watch"; Bat = "$repo\scripts\run_screener_watch_task.bat" },
  @{ Name = "NeuroTrade Williams Watch"; Bat = "$repo\scripts\run_williams_watch_task.bat" }
)

foreach ($t in $tasks) {
  $action  = New-ScheduledTaskAction -Execute "cmd.exe" `
                -Argument "/c `"$($t.Bat)`"" -WorkingDirectory $repo
  $trigger = New-ScheduledTaskTrigger -AtLogOn
  $principal = New-ScheduledTaskPrincipal -UserId $uid -LogonType Interactive
  $settings  = New-ScheduledTaskSettingsSet `
                -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
                -ExecutionTimeLimit ([TimeSpan]::Zero) `
                -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
                -MultipleInstances IgnoreNew
  Register-ScheduledTask -TaskName $t.Name -Action $action -Trigger $trigger `
                -Principal $principal -Settings $settings -Force | Out-Null
  Write-Host "registered: $($t.Name)" -ForegroundColor Green
}

Write-Host ""
Write-Host "Done. Starting them now so you don't have to log out/in:" -ForegroundColor Cyan
Start-ScheduledTask -TaskName "NeuroTrade Dashboard"
Start-ScheduledTask -TaskName "NeuroTrade Screener Watch"
Start-ScheduledTask -TaskName "NeuroTrade Williams Watch"

Get-ScheduledTask | Where-Object { $_.TaskName -like "*NeuroTrade*" } |
  Select-Object TaskName, State | Format-Table -AutoSize
