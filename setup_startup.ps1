# setup_startup.ps1 — Configure auto-login and launch.py on startup
# Run once as Administrator in PowerShell:
#   powershell -ExecutionPolicy Bypass -File setup_startup.ps1

$ErrorActionPreference = "Stop"

$username   = "Administrator"
$launchScript = "C:\Users\Administrator\IdeaProjects\HMM_XGBoost_H1_Swing\launch.py"
$python     = (Get-Command python.exe).Source
$taskName   = "HMM_XGBoost_H1_Swing"

# ── Prompt for password ────────────────────────────────────────────────────────
$securePass = Read-Host "Enter password for $username" -AsSecureString
$plainPass  = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
                [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePass))

# ── 1. Configure auto-login via registry ──────────────────────────────────────
Write-Host "`nConfiguring auto-login..."
$winlogon = "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon"
Set-ItemProperty $winlogon -Name "AutoAdminLogon"  -Value "1"
Set-ItemProperty $winlogon -Name "DefaultUserName"  -Value $username
Set-ItemProperty $winlogon -Name "DefaultPassword"  -Value $plainPass
Set-ItemProperty $winlogon -Name "DefaultDomainName" -Value $env:COMPUTERNAME
Write-Host "  Auto-login configured for $username"

# ── 2. Create scheduled task ───────────────────────────────────────────────────
Write-Host "`nCreating scheduled task '$taskName'..."

# Remove existing task if present
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

$action = New-ScheduledTaskAction `
    -Execute $python `
    -Argument "`"$launchScript`"" `
    -WorkingDirectory "C:\Users\Administrator\IdeaProjects\HMM_XGBoost_H1_Swing"

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
    -Description "Launch MT5 and HMM_XGBoost_H1_Swing live trader on startup"

Write-Host "  Task '$taskName' created."
Write-Host "`nDone. Reboot to test - MT5 and the live trader will start automatically."
