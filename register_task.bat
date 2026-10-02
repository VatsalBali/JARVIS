@echo off
REM Register a scheduled task to run log_ram.ps1 every hour
PowerShell -NoProfile -ExecutionPolicy Bypass -Command "$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-NoProfile -WindowStyle Hidden -File \"%~dp0log_ram.ps1\"'; $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Hours 1) -RepetitionDuration ([TimeSpan]::MaxValue); Register-ScheduledTask -TaskName 'LogRAMUsage' -Action $action -Trigger $trigger -Description 'Logs RAM usage every hour' -User $(whoami) -RunLevel Highest -Force"
