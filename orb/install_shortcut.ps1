# Creates ORACLE shortcuts in the Start menu (with the global hotkey
# Ctrl+Alt+O) and on the desktop. Windows only honours shortcut hotkeys for
# Start-menu/desktop shortcuts, and two with the same hotkey conflict, so
# only the Start-menu one has it. Starting ORACLE while it runs just talks.
#
#   powershell -ExecutionPolicy Bypass -File install_shortcut.ps1

$orb = $PSScriptRoot
$electron = Join-Path $orb 'node_modules\electron\dist\electron.exe'
if (-not (Test-Path $electron)) { throw "Electron not found at $electron - run 'npm install' in $orb first." }
$icon = Join-Path (Split-Path $orb -Parent) 'jarvis.ico'
$shell = New-Object -ComObject WScript.Shell

function New-OracleShortcut($folder, $hotkey) {
    $link = Join-Path $folder 'ORACLE.lnk'
    $sc = $shell.CreateShortcut($link)
    $sc.TargetPath = $electron
    $sc.Arguments = "`"$orb`""
    $sc.WorkingDirectory = $orb
    if ($hotkey) { $sc.Hotkey = $hotkey }
    $sc.WindowStyle = 7  # minimised: no flash of a window while it starts
    $sc.Description = 'ORACLE voice assistant'
    if (Test-Path $icon) { $sc.IconLocation = $icon }
    $sc.Save()
    Write-Output "Created $link$(if ($hotkey) { " (hotkey $hotkey)" })"
}

New-OracleShortcut ([Environment]::GetFolderPath('Programs')) 'CTRL+ALT+O'
New-OracleShortcut ([Environment]::GetFolderPath('Desktop')) $null
