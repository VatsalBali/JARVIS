# Creates a Start-menu shortcut for ORACLE with the global hotkey Ctrl+Alt+O.
# Windows only honours shortcut hotkeys for Start-menu/desktop shortcuts.
# Pressing it starts ORACLE; while ORACLE runs, the same keys start talking.
#
#   powershell -ExecutionPolicy Bypass -File install_shortcut.ps1

$orb = $PSScriptRoot
$electron = Join-Path $orb 'node_modules\electron\dist\electron.exe'
if (-not (Test-Path $electron)) { throw "Electron not found at $electron - run 'npm install' in $orb first." }

$link = Join-Path ([Environment]::GetFolderPath('Programs')) 'ORACLE.lnk'
$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($link)
$sc.TargetPath = $electron
$sc.Arguments = "`"$orb`""
$sc.WorkingDirectory = $orb
$sc.Hotkey = 'CTRL+ALT+O'
$sc.WindowStyle = 7  # minimised: no flash of a window while it starts
$sc.Description = 'ORACLE voice assistant'
$icon = Join-Path (Split-Path $orb -Parent) 'jarvis.ico'
if (Test-Path $icon) { $sc.IconLocation = $icon }
$sc.Save()
Write-Output "Created $link (hotkey Ctrl+Alt+O)"
