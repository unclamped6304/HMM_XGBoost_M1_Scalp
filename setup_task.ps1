# setup_task.ps1 — Register the HMM_XGBoost_M15_TrendFollow scheduled task
# Run once as Administrator in PowerShell:
#   powershell -ExecutionPolicy Bypass -File setup_task.ps1

$ErrorActionPreference = "Stop"

$username     = "Administrator"
$launchScript = "C:\Users\Administrator\IdeaProjects\HMM_XGBoost_M15_Scalp\launch.py"
$python       = (Get-Command python.exe).Source
$taskName     = "HMM_XGBoost_M15_TrendFollow"

# ── Create scheduled task ──────────────────────────────────────────────────────
Write-Host "Creating scheduled task '$taskName'..."

# Remove existing task if present
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

$action = New-ScheduledTaskAction `
    -Execute $python `
    -Argument "`"$launchScript`"" `
    -WorkingDirectory "C:\Users\Administrator\IdeaProjects\HMM_XGBoost_M15_Scalp"

# Trigger: at logon of this user, with 30s delay to let desktop settle
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $username
$trigger.Delay = "PT30S"

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -StartWhenAvailable

$principal = New-ScheduledTaskPrincipal `
    -UserId $username `
    -LogonType Interactive `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName   $taskName `
    -Action     $action `
    -Trigger    $trigger `
    -Settings   $settings `
    -Principal  $principal `
    -Description "Launch MT5 and HMM_XGBoost_M15_TrendFollow live trader on startup"

Write-Host "  Task '$taskName' created."
Write-Host "`nDone. The live trader will launch automatically at next logon."
