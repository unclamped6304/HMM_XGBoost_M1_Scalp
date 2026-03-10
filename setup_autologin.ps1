# setup_autologin.ps1 — Configure auto-login for Administrator
# Run once as Administrator in PowerShell:
#   powershell -ExecutionPolicy Bypass -File setup_autologin.ps1

$ErrorActionPreference = "Stop"

$username = "Administrator"

# ── Prompt for password ────────────────────────────────────────────────────────
$securePass = Read-Host "Enter password for $username" -AsSecureString
$plainPass  = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
                [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePass))

# ── Configure auto-login via registry ─────────────────────────────────────────
Write-Host "`nConfiguring auto-login..."
$winlogon = "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon"
Set-ItemProperty $winlogon -Name "AutoAdminLogon"   -Value "1"
Set-ItemProperty $winlogon -Name "DefaultUserName"  -Value $username
Set-ItemProperty $winlogon -Name "DefaultPassword"  -Value $plainPass
Set-ItemProperty $winlogon -Name "DefaultDomainName" -Value $env:COMPUTERNAME
Write-Host "  Auto-login configured for $username"

Write-Host "`nDone. Reboot to verify auto-login is working."
