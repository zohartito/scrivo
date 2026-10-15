# Install Scrivo on Windows with one line (in PowerShell):
#
#   irm https://raw.githubusercontent.com/zohartito/scrivo/main/scripts/get.ps1 | iex
#
# Downloads Scrivo with git into %USERPROFILE%\scrivo (or $env:SCRIVO_SOURCE; a
# copy installed before the rename, in %USERPROFILE%\sotto, stays there), or
# updates that copy (the new packages first, the source only after them), then runs its
# scripts\install-windows.ps1 and starts Scrivo.
# No admin rights. Missing tools are named with the command that installs them.

# A folder has a long name and, on most volumes, an 8.3 short one
# (C:\Users\RUNNER~1); a running process can use either. Paths are
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

# Same repository? Ignores https vs either ssh form (git@github.com:... and
# ssh://git@github.com/...), a trailing .git and a trailing slash.
function Test-SameRepo([string]$First, [string]$Second) {
  $normalized = foreach ($url in $First, $Second) {
    $https = $url -replace '^git@github\.com:', 'https://github.com/' -replace '^ssh://git@github\.com/', 'https://github.com/'
    ($https.TrimEnd('/')) -replace '\.git$', ''
  }
  return $normalized[0] -eq $normalized[1]
}

# The one-liner runs this script inside the user's shell (iex), where exit
# would close that shell: a failure is said, remembered here, and becomes an
# exit code only when the script runs as a file.
function Fail-Install([string]$Text) {
  Write-Host "X $Text" -ForegroundColor Red
  $script:ScrivoInstallFailed = $true
}

function Install-Scrivo {
  $ErrorActionPreference = 'Stop'
  # This project, and its URL from before the rename (Scrivo was called
  # Sotto; SOTTO_* is the old name of each SCRIVO_* variable).
  $scrivoRepo = 'https://github.com/zohartito/scrivo.git'
  $legacyRepo = 'https://github.com/zohartito/sotto.git'
  $repo = if ($env:SCRIVO_REPO) { $env:SCRIVO_REPO } elseif ($env:SOTTO_REPO) { $env:SOTTO_REPO } else { $scrivoRepo }
  $dest = if ($env:SCRIVO_SOURCE) { $env:SCRIVO_SOURCE } elseif ($env:SOTTO_SOURCE) { $env:SOTTO_SOURCE } else { '' }
  $shortcutDir = [Environment]::GetFolderPath('Programs')
  # A copy of this project: it follows $repo or, for this project, the URL
  # from before the rename (GitHub redirects it).
  function Test-FollowsThis([string]$Origin) {
    return (Test-SameRepo $Origin $repo) -or ((Test-SameRepo $repo $scrivoRepo) -and (Test-SameRepo $Origin $legacyRepo))
  }

  if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail-Install 'Scrivo needs Git: winget install --id Git.Git -e  (then open a new PowerShell and run this line again)'
    return
  }
  if (-not $dest) {
    $dest = Join-Path $env:USERPROFILE 'scrivo'
    # A copy installed before the rename is in %USERPROFILE%\sotto, and its
    # Start Menu entry and login entry run it by that path: update it where it
    # is, unless %USERPROFILE%\scrivo exists.
    $legacyDest = Join-Path $env:USERPROFILE 'sotto'
    if (-not (Test-Path -LiteralPath $dest) -and (Test-Path -LiteralPath (Join-Path $legacyDest '.git'))) {
      $legacyOrigin = ''
      try { $legacyOrigin = git -C $legacyDest remote get-url origin 2>$null } catch { $legacyOrigin = '' }
      if (Test-FollowsThis $legacyOrigin) { $dest = $legacyDest }
    }
  }
  if (Test-Path -LiteralPath (Join-Path $dest '.git')) {
    $origin = ''
    try { $origin = git -C $dest remote get-url origin 2>$null } catch { $origin = '' }
    $renamed = ''
    if (-not (Test-SameRepo $origin $repo)) {
      # A copy made before the rename follows the old URL: the same project,
      # moved to the new URL below.
      if (-not (Test-FollowsThis $origin)) {
        Fail-Install "$dest is a git copy of $origin, not Scrivo. Choose another folder with `$env:SCRIVO_SOURCE."
        return
      }
      $renamed = $origin -replace 'zohartito/sotto', 'zohartito/scrivo'  # same form: https or ssh
    }
    # Pulling and reinstalling under a running Scrivo would swap its source and
    # packages while they are in use.
    $prefix = (Long-Path $dest).TrimEnd('\') + '\'
    $running = @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $path = $null
        try { $path = $_.Path } catch { }
        $path -and (Long-Path $path).StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
      })
    if ($running.Count -gt 0) {
      Fail-Install "Scrivo is running from $dest (process $($running[0].Id)). Quit it from its tray menu (or update it there with Check for updates), then run this line again."
      return
    }
    if ($renamed) {
      git -C $dest remote set-url origin $renamed
      Write-Host "== $dest now follows $renamed (Sotto is now called Scrivo)"
    }
    Write-Host "== Updating Scrivo in $dest"
    git -C $dest fetch --quiet
    if ($LASTEXITCODE -ne 0) { Fail-Install "Could not download the update into $dest."; return }
    # Ask the update's own installer whether it will run here (its Start Menu
    # check) before the source moves: a refusal after the pull would leave new
    # source on old packages.
    $check = Join-Path ([IO.Path]::GetTempPath()) "scrivo-get-$PID"
    Remove-Item -LiteralPath $check -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $check | Out-Null
    try {
      $archive = Join-Path $check 'installer.tar'
      git -C $dest archive --format=tar -o $archive '@{u}' scripts/install-windows.ps1 scripts/update-windows.ps1
      if ($LASTEXITCODE -eq 0) { & (Join-Path $env:SystemRoot 'System32\tar.exe') -xf $archive -C $check }
      $installer = Join-Path $check 'scripts\install-windows.ps1'
      $updater = Join-Path $check 'scripts\update-windows.ps1'
      if (-not (Test-Path -LiteralPath $installer -PathType Leaf) -or -not (Test-Path -LiteralPath $updater -PathType Leaf)) {
        Fail-Install "Could not read the update's installer; $dest was not changed."; return
      }
      & powershell -NoProfile -ExecutionPolicy Bypass -File $installer -CheckOnly -SourceDir $dest -ShortcutDir $shortcutDir
      if ($LASTEXITCODE -ne 0) { Fail-Install "$dest was not updated; see the message above."; return }
      $behind = "$(git -C $dest rev-parse HEAD)".Trim() -ne "$(git -C $dest rev-parse '@{u}')".Trim()
      if ($behind -and (Test-Path -LiteralPath (Join-Path $dest 'venv-alpha\Scripts\python.exe') -PathType Leaf)) {
        # An installed copy: the update's own updater installs the new
        # packages first, moves the source only after them, and puts the
        # previous packages back if anything fails.
        & powershell -NoProfile -ExecutionPolicy Bypass -File $updater -SourceDir $dest -NoRestart `
          -WaitSeconds 0 -ShortcutDir $shortcutDir
        if ($LASTEXITCODE -ne 0) { Fail-Install "$dest was not updated; see the message above."; return }
      }
    } finally {
      Remove-Item -LiteralPath $check -Recurse -Force -ErrorAction SilentlyContinue
    }
    # Already moved by the updater, or nothing installed yet: no packages to keep in step.
    git -C $dest merge --ff-only --quiet '@{u}'
    if ($LASTEXITCODE -ne 0) { Fail-Install "Could not update $dest (local changes?)."; return }
  } elseif (Test-Path -LiteralPath $dest) {
    Fail-Install "$dest exists and is not a copy of Scrivo. Choose another folder with `$env:SCRIVO_SOURCE."
    return
  } else {
    Write-Host "== Downloading Scrivo into $dest"
    git clone --quiet $repo $dest
    if ($LASTEXITCODE -ne 0) { Fail-Install 'Could not download Scrivo.'; return }
  }

  if ($env:SCRIVO_GET_DRY_RUN) { "dry run: would run $dest\scripts\install-windows.ps1"; return }
  # The installer checks Windows, Python 3.13 and the GPU, and says what is missing.
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $dest 'scripts\install-windows.ps1') -ShortcutDir $shortcutDir
  if ($LASTEXITCODE -ne 0) { Fail-Install 'The installer stopped; see the messages above.'; return }
  $shortcut = Join-Path $shortcutDir 'Scrivo.lnk'
  if (Test-Path -LiteralPath $shortcut) { Start-Process -FilePath $shortcut }
}

$script:ScrivoInstallFailed = $false
Install-Scrivo
if ($script:ScrivoInstallFailed) {
  $global:LASTEXITCODE = 1
  # Only when run as a file (powershell -File get.ps1): a function defined from
  # text pasted through iex has no file, and there exit would close the shell.
  if (${function:Install-Scrivo}.File) { exit 1 }
} else {
  $global:LASTEXITCODE = 0
}
