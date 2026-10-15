<#
.SYNOPSIS
  Install (or remove) the Scrivo Windows source alpha for the current user.

.DESCRIPTION
  Checks for Windows x64 and Python 3.13 (64-bit), creates venv-alpha in this
  folder, installs the pinned CPU or CUDA dependency set (CUDA when
  nvidia-smi sees a GPU; a re-run or update keeps the set recorded at
  install; -Cpu / -Cuda override), prepares the alpha data
  folder, downloads the pinned speech model and voice detection into it
  (scrivo_win.py setup; -SkipSetup leaves that to the first launch), and adds a
  Start Menu shortcut "Scrivo" that starts the tray app without a console
  window, with SCRIVO_DATA_DIR applied.

  Nothing is installed system-wide: no admin rights, services or scheduled
  tasks. Python itself is never installed for you; the script says exactly
  what to install instead. Re-running is safe. It refuses to run while Scrivo
  runs from this environment (Windows would swap its packages underneath it)
  and never replaces a Start Menu "Scrivo" shortcut that starts another copy;
  the "Sotto" entry of earlier versions (Scrivo's former name) is removed
  when it starts this copy. A source archive (folder scrivo-alpha) extracted
  next to an earlier one (folder sotto-alpha) takes over that copy's Start
  Menu entry and launch-at-login entry.
  -Uninstall removes only what
  this script recorded creating (and the login entry the Settings window may
  have added); your data folder stays unless you add -RemoveData.
  -CheckOnly runs only the running-Scrivo and shortcut checks, for the copy in -SourceDir
  (default: this one), and changes nothing; get.ps1 asks the update's
  installer this before it moves an existing copy's source.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install-windows.ps1
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install-windows.ps1 -WhatIf
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install-windows.ps1 -Uninstall
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
  [switch]$Cpu,
  [switch]$Cuda,
  [string]$DataDir = '',
  [string]$Python = '',
  [string]$VenvDir = '',
  [string]$ShortcutDir = '',
  [switch]$SkipSetup,
  [switch]$Uninstall,
  [switch]$RemoveData,
  [switch]$CheckOnly,
  [string]$SourceDir = '',
  # Tests point this at a throwaway key; the default is the per-user login list.
  [string]$RunKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run'
)

Set-StrictMode -Version 2
$ErrorActionPreference = 'Stop'

# A folder has a long name and, on most volumes, an 8.3 short one
# (C:\Users\RUNNER~1); a process, shortcut or record can use either. Paths are
# compared by their long names: Windows PowerShell's GetFullPath expands only
# a path that exists and PowerShell 7's never does, so the part that exists
# goes through GetLongPathName.
if (-not ('ScrivoPaths.Native' -as [type])) {
  Add-Type -Namespace ScrivoPaths -Name Native -MemberDefinition @'
[DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
public static extern uint GetLongPathNameW(string shortPath, System.Text.StringBuilder longPath, uint size);
'@
}
function Long-Path([string]$Path) {
  $full = [IO.Path]::GetFullPath($Path)
  if ($full.Length -gt 3) { $full = $full.TrimEnd('\') }
  $existing, $rest = $full, ''
  while ($existing -and -not ([IO.Directory]::Exists($existing) -or [IO.File]::Exists($existing))) {
    $rest = '\' + [IO.Path]::GetFileName($existing) + $rest
    $existing = [IO.Path]::GetDirectoryName($existing)
  }
  $buffer = New-Object Text.StringBuilder 32768
  if (-not $existing -or [ScrivoPaths.Native]::GetLongPathNameW($existing, $buffer, $buffer.Capacity) -eq 0) { return $full }
  if ($rest) { return $buffer.ToString().TrimEnd('\') + $rest }
  return $buffer.ToString()
}
function Same-Path([string]$First, [string]$Second) {
  return [bool]($First -and $Second) -and [string]::Equals((Long-Path $First), (Long-Path $Second), [StringComparison]::OrdinalIgnoreCase)
}

$Root = Long-Path (Split-Path -Parent $PSScriptRoot)
if ($SourceDir) {
  if (-not $CheckOnly) { Write-Host 'X -SourceDir goes with -CheckOnly.' -ForegroundColor Red; exit 1 }
  $Root = Long-Path $SourceDir
}
$Launcher = Join-Path $Root 'win_launch.py'
if (-not $DataDir) { $DataDir = Join-Path $env:LOCALAPPDATA 'sotto-alpha' }
if (-not $VenvDir) { $VenvDir = Join-Path $Root 'venv-alpha' }
if (-not $ShortcutDir) { $ShortcutDir = [Environment]::GetFolderPath('Programs') }
$DataDir = Long-Path $DataDir
$VenvDir = Long-Path $VenvDir
$Shortcut = Join-Path $ShortcutDir 'Scrivo.lnk'
# Scrivo was called Sotto: the entry earlier versions made, replaced when it starts this copy.
$LegacyShortcut = Join-Path $ShortcutDir 'Sotto.lnk'
$Manifest = Join-Path $VenvDir 'scrivo-install.json'
$LegacyManifest = Join-Path $VenvDir 'sotto-install.json'  # the record's name before the rename
$LegacyData = Join-Path $env:APPDATA 'sotto'
# The login entry keeps the value name 'Sotto' from before the rename.
$RunValue = 'Sotto'
# A source archive extracted next to the archive of an earlier version: that
# copy's launcher (Scrivo was called Sotto, and so was the archive's folder).
$EarlierLauncher = ''
if ((Split-Path -Leaf $Root) -eq 'scrivo-alpha') {
  $EarlierLauncher = Join-Path (Split-Path -Parent $Root) 'sotto-alpha\win_launch.py'
}

function Say([string]$Text) { Write-Host $Text }
# Windows PowerShell 5.1 turns redirected native stderr into terminating
# errors under 'Stop'; probes that may fail run through this instead.
function Quiet([scriptblock]$Block) {
  $saved = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try { & $Block 2>$null } finally { $ErrorActionPreference = $saved }
}
function Fail([string]$Text) { Write-Host ''; Write-Host "X $Text" -ForegroundColor Red; exit 1 }
# Long native steps (venv, pip): their warnings on stderr must stay warnings.
function Native([scriptblock]$Block) {
  $saved = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try { & $Block } finally { $ErrorActionPreference = $saved }
}

function Read-Manifest {
  foreach ($path in $Manifest, $LegacyManifest) {
    if (Test-Path -LiteralPath $path -PathType Leaf) {
      return Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
    }
  }
  return $null
}

function Write-Manifest([bool]$CreatedVenv, [bool]$CreatedData, [string]$Flavor, [string]$Python) {
  if ($PSCmdlet.ShouldProcess($Manifest, 'Record what this script created')) {
    [ordered]@{
      schema = 1; created_venv = $CreatedVenv; created_data = $CreatedData; data_dir = $DataDir
      shortcut = $Shortcut; flavor = $Flavor; python = $Python
    } | ConvertTo-Json | Set-Content -LiteralPath $Manifest -Encoding UTF8
    Remove-Item -LiteralPath $LegacyManifest -Force -ErrorAction SilentlyContinue  # its content is in $Manifest now
  }
}

# Does this command line (a shortcut's arguments, a login entry) start this
# launcher? Its paths are quoted only when they contain a space.
function Points-At([string]$Command, [string]$Target) {
  foreach ($word in [regex]::Matches("$Command", '"([^"]*)"|(\S+)')) {
    $path = if ($word.Groups[1].Success) { $word.Groups[1].Value } else { $word.Groups[2].Value }
    if ($path -match '^([A-Za-z]:\\|\\\\)' -and (Same-Path $path $Target)) { return $true }
  }
  return $false
}
function Points-Here([string]$Command) { return Points-At $Command $Launcher }
# ... or the earlier archive's copy this one replaces.
function Points-Earlier([string]$Command) { return [bool]$EarlierLauncher -and (Points-At $Command $EarlierLauncher) }

# The Start Menu entry of earlier versions ("Sotto"), only when it starts this
# copy (or, installing, the earlier archive's copy this one replaces).
function Remove-LegacyShortcut([bool]$Earlier = $false) {
  if (-not (Test-Path -LiteralPath $LegacyShortcut -PathType Leaf)) { return }
  $link = (New-Object -ComObject WScript.Shell).CreateShortcut($LegacyShortcut)
  $ours = (Points-Here $link.Arguments) -or ($Earlier -and (Points-Earlier $link.Arguments))
  if ($ours -and $PSCmdlet.ShouldProcess($LegacyShortcut, 'Remove the Start Menu shortcut "Sotto" of earlier versions')) {
    Remove-Item -LiteralPath $LegacyShortcut -Force
    Say "removed the old Start Menu shortcut $LegacyShortcut (Sotto is now called Scrivo)"
  }
}

function Running-From-Venv {
  $prefix = $VenvDir.TrimEnd('\') + '\'
  return @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
      $path = $null
      try { $path = $_.Path } catch { }
      $path -and (Long-Path $path).StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
    })
}

# ---------------------------------------------------------------- uninstall
if ($Uninstall) {
  $record = Read-Manifest
  $running = @(Running-From-Venv)  # an empty result would otherwise unroll to $null
  if ($running.Count -gt 0) {
    Fail "Scrivo is still running from $VenvDir (process $($running[0].Id)). Quit it from its tray menu, then run -Uninstall again."
  }
  if (Test-Path -LiteralPath $Shortcut -PathType Leaf) {
    $link = (New-Object -ComObject WScript.Shell).CreateShortcut($Shortcut)
    if (Points-Here $link.Arguments) {
      if ($PSCmdlet.ShouldProcess($Shortcut, 'Remove Start Menu shortcut')) {
        Remove-Item -LiteralPath $Shortcut -Force
        Say "removed Start Menu shortcut $Shortcut"
      }
    } else { Say "kept $Shortcut (it does not start this copy of Scrivo)" }
  }
  Remove-LegacyShortcut
  $run = Get-ItemProperty -Path $RunKey -Name $RunValue -ErrorAction SilentlyContinue
  if ($run -and (Points-Here $run.$RunValue)) {
    if ($PSCmdlet.ShouldProcess("$RunKey\$RunValue", 'Remove launch-at-login entry')) {
      Remove-ItemProperty -Path $RunKey -Name $RunValue
      Say 'removed the launch-at-login entry'
    }
  }
  $isVenv = Test-Path -LiteralPath (Join-Path $VenvDir 'pyvenv.cfg') -PathType Leaf
  if ($record -and $record.created_venv -and $isVenv) {
    if ($PSCmdlet.ShouldProcess($VenvDir, 'Remove the Python environment')) {
      Remove-Item -LiteralPath $VenvDir -Recurse -Force
      Say "removed $VenvDir"
    }
  } elseif (Test-Path -LiteralPath $VenvDir) {
    Say "kept $VenvDir (this script did not create it)"
  }
  if ($RemoveData) {
    if (-not ($record -and $record.created_data -and (Same-Path $record.data_dir $DataDir))) {
      Say "kept $DataDir (this script did not create it; delete it yourself if you want)"
    } elseif (Same-Path $DataDir $LegacyData) {
      Say "kept $DataDir (never removed automatically)"
    } elseif ((Test-Path -LiteralPath $DataDir) -and $PSCmdlet.ShouldProcess($DataDir, 'Remove Scrivo data: history, recordings, models, dictionary')) {
      Remove-Item -LiteralPath $DataDir -Recurse -Force
      Say "removed $DataDir"
    }
  } elseif (Test-Path -LiteralPath $DataDir) {
    Say "kept your data in $DataDir (history, recordings, models, dictionary); add -RemoveData to delete it"
  }
  Say 'uninstall finished'
  exit 0
}

# ---------------------------------------------------------------- checks
if ($Cpu -and $Cuda) { Fail 'Choose at most one of -Cpu and -Cuda.' }
$running = @(Running-From-Venv)
if ($running.Count -gt 0) {
  Fail "Scrivo is still running from $VenvDir (process $($running[0].Id)); installing now would replace its packages while they are in use. Quit it from its tray menu (or update it there with Check for updates), then run this again."
}
# Never take over a Start Menu entry that starts another copy of Scrivo (or
# another program). One whose copy no longer exists is replaced.
if (Test-Path -LiteralPath $Shortcut -PathType Leaf) {
  $existing = (New-Object -ComObject WScript.Shell).CreateShortcut($Shortcut)
  if (-not (Points-Here $existing.Arguments) -and -not (Points-Earlier $existing.Arguments)) {
    $gone = ($existing.Arguments -match '^\s*"([^"]*win_launch\.py)"') -and -not (Test-Path -LiteralPath $Matches[1])
    if (-not $gone) {
      Fail "$Shortcut already starts another copy of Scrivo or another program ($($existing.TargetPath) $($existing.Arguments)). Remove that copy with its own install-windows.ps1 -Uninstall, or pass -ShortcutDir with another folder."
    }
    Say "! $Shortcut started a copy of Scrivo that no longer exists; it will start this one"
  }
}
if ($CheckOnly) { Say "OK nothing stops installing Scrivo in $Root"; exit 0 }
$arch = $env:PROCESSOR_ARCHITEW6432
if (-not $arch) { $arch = $env:PROCESSOR_ARCHITECTURE }
if (-not [Environment]::Is64BitOperatingSystem -or $arch -ne 'AMD64') {
  Fail "Scrivo's Windows alpha supports Windows x64 only (this PC reports $arch)."
}
Say "OK Windows x64 ($([Environment]::OSVersion.VersionString))"

# No double quotes inside: Windows PowerShell 5.1 does not escape them for native commands.
$probe = "import struct, sys; print(sys.executable); print('%d.%d.%d' % sys.version_info[:3]); print(struct.calcsize('P') * 8)"
function Probe-Python([string[]]$Command) {
  try {
    $exe = $Command[0]
    $rest = @()
    if ($Command.Count -gt 1) { $rest = $Command[1..($Command.Count - 1)] }
    $out = Quiet { & $exe @rest -c $probe }
    if ($LASTEXITCODE -ne 0 -or -not $out -or @($out).Count -lt 3) { return $null }
    return @{ Exe = $out[0]; Version = $out[1]; Bits = $out[2] }
  } catch { return $null }
}
if ($Python) { $found = Probe-Python @($Python) } else { $found = Probe-Python @('py', '-3.13') }
if (-not $found -and -not $Python) { $found = Probe-Python @('python') }
$instructions = "Install Python 3.13 (64-bit) from https://www.python.org/downloads/windows/ " +
  "(the 'Windows installer (64-bit)'), keep 'py launcher' and 'tcl/tk and IDLE' ticked, " +
  "then run this script again. Or pass -Python C:\path\to\python.exe."
if (-not $found) { Fail "Python 3.13 was not found. $instructions" }
if (-not $found.Version.StartsWith('3.13.') -or $found.Bits -ne '64') {
  Fail "Found Python $($found.Version) ($($found.Bits)-bit) at $($found.Exe); Scrivo needs 3.13 64-bit. $instructions"
}
$BasePython = $found.Exe
Say "OK Python $($found.Version) 64-bit ($BasePython)"
Quiet { & $BasePython -c 'import tkinter' } | Out-Null
if ($LASTEXITCODE -ne 0) {
  Say "! This Python has no tkinter: the tray works, but Settings and Correct need it. Re-run the python.org installer, choose Modify, tick 'tcl/tk and IDLE'."
}

$record = Read-Manifest
$recorded = ''
if ($record -and $record.PSObject.Properties['flavor'] -and @('cpu', 'cuda') -contains $record.flavor) {
  $recorded = $record.flavor
}
if ($Cuda) { $flavor = 'cuda'; $why = '-Cuda' }
elseif ($Cpu) { $flavor = 'cpu'; $why = '-Cpu' }
elseif ($recorded) { $flavor = $recorded; $why = 'recorded at install; -Cpu or -Cuda changes it' }
else {
  $flavor = 'cpu'; $why = 'no NVIDIA GPU found'
  if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpus = @(Quiet { & nvidia-smi -L } | Where-Object { $_ -like 'GPU *' })
    if ($LASTEXITCODE -eq 0 -and $gpus.Count -gt 0) { $flavor = 'cuda'; $why = "nvidia-smi: $($gpus[0])" }
  }
}
if ($flavor -eq 'cuda') { $Requirements = Join-Path $Root 'requirements-alpha-windows-cuda.txt' }
else { $Requirements = Join-Path $Root 'requirements-alpha-windows.txt' }
Say "OK dependency set: $flavor ($why)"

# ---------------------------------------------------------------- install
$createdVenv = [bool]($record -and $record.created_venv)
$createdData = [bool]($record -and $record.created_data -and (Same-Path $record.data_dir $DataDir))
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
$VenvPythonW = Join-Path $VenvDir 'Scripts\pythonw.exe'

if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
  $existing = Probe-Python @($VenvPython)
  if (-not $existing -or -not $existing.Version.StartsWith('3.13.')) {
    Fail "$VenvDir holds a different Python; remove it (or pass -VenvDir) and run again."
  }
  Say "OK reusing $VenvDir"
} elseif ((Test-Path -LiteralPath $VenvDir) -and @(Get-ChildItem -LiteralPath $VenvDir -Force).Count -gt 0) {
  # Never adopt (and later delete) a folder this script did not make.
  Fail "$VenvDir already exists but is not a Windows Python environment; pass -VenvDir with a new folder."
} elseif ($PSCmdlet.ShouldProcess($VenvDir, 'Create Python environment')) {
  Native { & $BasePython -m venv $VenvDir }
  if ($LASTEXITCODE -ne 0) { Fail "Could not create $VenvDir." }
  $createdVenv = $true
  Write-Manifest $createdVenv $createdData $flavor $BasePython  # ownership survives a failed pip run
  Say "OK created $VenvDir"
}

if ($PSCmdlet.ShouldProcess($VenvDir, "Install pinned packages from $(Split-Path -Leaf $Requirements)")) {
  # An exact pip, not whichever one this Python shipped (F25); tests/test_dependency_pins.py keeps every copy in step.
  Native { & $VenvPython -m pip install --disable-pip-version-check pip==26.2.1 }
  if ($LASTEXITCODE -ne 0) { Fail 'Could not install pip 26.2.1 (see above); check the network and run again.' }
  Native { & $VenvPython -m pip install --disable-pip-version-check -r $Requirements }
  if ($LASTEXITCODE -ne 0) { Fail 'pip could not install the pinned packages (see above); check the network and run again.' }
  Native { & $VenvPython -m pip check --disable-pip-version-check }
  if ($LASTEXITCODE -ne 0) { Fail 'pip check found broken requirements (see above).' }
  Say 'OK packages installed and consistent'
}

if (Test-Path -LiteralPath $DataDir) {
  Say "OK data folder $DataDir"
} elseif ($PSCmdlet.ShouldProcess($DataDir, 'Create the alpha data folder')) {
  New-Item -ItemType Directory -Path $DataDir | Out-Null
  $createdData = $true
  Say "OK created data folder $DataDir"
}

if (-not $SkipSetup -and $PSCmdlet.ShouldProcess($DataDir, 'Download the pinned speech model (about 1.6 GB, once) and voice detection')) {
  # Set only for the setup child and put back afterwards: a script run from an
  # interactive shell must not leave them in that shell.
  $savedData, $savedHf = $env:SCRIVO_DATA_DIR, $env:SCRIVO_HF_HOME
  try {
    $env:SCRIVO_DATA_DIR = $DataDir
    $env:SCRIVO_HF_HOME = Join-Path $DataDir 'huggingface'
    Native { & $VenvPython (Join-Path $Root 'scrivo_win.py') setup }
    $setupExit = $LASTEXITCODE
  } finally {
    $env:SCRIVO_DATA_DIR, $env:SCRIVO_HF_HOME = $savedData, $savedHf
  }
  if ($setupExit -ne 0) {
    Say '! The model download did not finish; Scrivo downloads it on first launch (or run this script again).'
  } elseif (Test-Path -LiteralPath (Join-Path $DataDir 'models\silero_vad.onnx') -PathType Leaf) {
    Say 'OK speech model and voice detection ready'
  } else {
    Say '! Speech model ready, but voice detection was not installed; run this script again later.'
  }
}

if ($PSCmdlet.ShouldProcess($Shortcut, 'Create Start Menu shortcut "Scrivo" (tray app, no console)')) {
  if (-not (Test-Path -LiteralPath $ShortcutDir)) { New-Item -ItemType Directory -Path $ShortcutDir | Out-Null }
  $link = (New-Object -ComObject WScript.Shell).CreateShortcut($Shortcut)
  $link.TargetPath = $VenvPythonW
  $link.Arguments = "`"$Launcher`" --data-dir `"$DataDir`""
  $link.WorkingDirectory = $Root
  $link.Description = 'Scrivo: local push-to-talk dictation'
  $link.Save()
  Say "OK Start Menu shortcut $Shortcut"
}
Remove-LegacyShortcut $true
# Launch at login of the earlier archive's copy: now this copy, with the
# command the Settings window writes (win_startup.launch_command).
$run = Get-ItemProperty -Path $RunKey -Name $RunValue -ErrorAction SilentlyContinue
if ($run -and (Points-Earlier $run.$RunValue) -and $RunKey -match '^HKCU:\\(.+)$' -and
    $PSCmdlet.ShouldProcess("$RunKey\$RunValue", 'Start this copy at login instead of the earlier one')) {
  $subkey = $Matches[1]
  $relink = "import sys; sys.path.insert(0, sys.argv[1]); from pathlib import Path; import win_startup; " +
    "win_startup.set_enabled(True, win_startup.launch_command(Path(sys.argv[2]), Path(sys.argv[3])), key=sys.argv[4])"
  Native { & $VenvPython -c $relink $Root $DataDir (Join-Path $DataDir 'huggingface') $subkey }
  if ($LASTEXITCODE -eq 0) { Say 'OK launch at login now starts this copy' }
  else {
    # Never leave it starting the earlier copy (whose single-instance lock would then keep this one out).
    Remove-ItemProperty -Path $RunKey -Name $RunValue
    Say '! Launch at login started the earlier copy and could not be moved here (see above), so it is off; turn it on in Settings.'
  }
}

Write-Manifest $createdVenv $createdData $flavor $BasePython

Say ''
Say 'Scrivo is installed. Start it from the Start Menu: Scrivo (a tray icon appears, then a'
Say '"Ready" notification; if the model was not downloaded above, the first launch does it).'
Say 'Settings (tray menu) can turn on launch at login.'
Say 'Console run and checks, in PowerShell:'
Say "  `$env:SCRIVO_DATA_DIR = `"$DataDir`"; `$env:SCRIVO_HF_HOME = `"$DataDir\huggingface`""
Say "  & `"$VenvPython`" `"$(Join-Path $Root 'scrivo_win.py')`" doctor"
Say "  & `"$VenvPython`" `"$(Join-Path $Root 'scrivo_win.py')`""
Say "Remove: powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Uninstall"
exit 0
