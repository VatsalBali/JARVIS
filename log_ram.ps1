# PowerShell script to log RAM usage every hour
$logPath = "$env:USERPROFILE\Documents\ram_usage_log.txt"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$ram = Get-CimInstance Win32_OperatingSystem | Select-Object -ExpandProperty TotalVisibleMemorySize
$free = Get-CimInstance Win32_OperatingSystem | Select-Object -ExpandProperty FreePhysicalMemory
$usedGB = [math]::Round((($ram - $free) * 1KB) / 1GB, 2)
$totalGB = [math]::Round(($ram * 1KB) / 1GB, 2)
$percent = [math]::Round((($ram - $free) / $ram) * 100, 1)
$entry = "$timestamp - RAM Used: $usedGB GB / $totalGB GB ($percent%)"
Add-Content -Path $logPath -Value $entry
