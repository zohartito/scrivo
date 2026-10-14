"""Scrivo.app's identity, and the one-time move from the Sotto.app of
earlier versions (macOS).

Scrivo was called Sotto. Copies installed before the rename have a
``Sotto.app`` (bundle id ``org.sotto.alpha``) and maybe a login item with that
label. Its compiled launcher starts this folder's sotto.py, which never moves,
so the first start after an update can still come from the old app: then
``hand_off`` opens Scrivo.app instead, and Scrivo.app's own start
(``retire_legacy``) removes the old app and moves the login item over. Only an
old app built by this installer for this very folder is ever touched.
"""
from __future__ import annotations

import os
from pathlib import Path
import plistlib
import shutil
import subprocess

APP_NAME = "Scrivo.app"
EXECUTABLE = "Scrivo"
BUNDLE_ID = "io.github.zohartito.scrivo"
LEGACY_APP_NAME = "Sotto.app"
LEGACY_EXECUTABLE = "Sotto"
LEGACY_BUNDLE_ID = "org.sotto.alpha"
OPEN_TIMEOUT_S = 30


def candidate_folders(home: Path | None = None) -> list[Path]:
    """Where the installer puts the app: /Applications, else ~/Applications."""
    return [Path("/Applications"), (home or Path.home()) / "Applications"]


def bundle_info(app: Path) -> dict:
    """The app's Info.plist, or {} when it is missing or unreadable."""
    try:
        info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    except (OSError, plistlib.InvalidFileException, ValueError):
        return {}
    return info if isinstance(info, dict) else {}


def launcher_runs(app: Path, script: Path) -> bool:
    """Does the app's compiled launcher start this script? It holds the
    script's absolute path as one C string (scripts/install_app.py)."""
    name = bundle_info(app).get("CFBundleExecutable")
    if not isinstance(name, str) or not name or "/" in name:
        return False
    try:
        binary = (app / "Contents" / "MacOS" / name).read_bytes()
    except OSError:
        return False
    return b"\0" + os.fsencode(str(script)) + b"\0" in binary


def is_legacy_copy(app: Path, script: Path) -> bool:
    """An old Sotto.app this installer built for this folder."""
    return (app.name == LEGACY_APP_NAME and bundle_info(app).get("CFBundleIdentifier") == LEGACY_BUNDLE_ID
            and launcher_runs(app, script))


def legacy_copies(script: Path, folders: list[Path] | None = None) -> list[Path]:
    return [app for app in ((folder / LEGACY_APP_NAME) for folder in (folders or candidate_folders()))
            if is_legacy_copy(app, script)]


def installed_app(script: Path, folders: list[Path] | None = None) -> Path | None:
    """This folder's Scrivo.app, if the installer built one."""
    for folder in folders or candidate_folders():
        app = folder / APP_NAME
        if bundle_info(app).get("CFBundleIdentifier") == BUNDLE_ID and launcher_runs(app, script):
            return app
    return None


def started_by_legacy_app(app_executable: str) -> bool:
    """Was this process started by an old Sotto.app's launcher?"""
    executable = Path(app_executable.strip()) if app_executable.strip() else None
    return (executable is not None and executable.name == LEGACY_EXECUTABLE
            and len(executable.parents) > 2 and executable.parents[2].name == LEGACY_APP_NAME)


def hand_off(app_executable: str, script: Path, *, folders: list[Path] | None = None,
             run=subprocess.run) -> Path | None:
    """Started by an old Sotto.app (after Update Now, or by its login item):
    open this folder's Scrivo.app instead. Returns the app it opened, or None
    when this start should carry on (not the old app, or no Scrivo.app to
    open, or opening it failed), so the user is never left with nothing.

    Side effects: runs /usr/bin/open on Scrivo.app (it only brings an already
    running Scrivo to the front)."""
    if not started_by_legacy_app(app_executable):
        return None
    target = installed_app(script, folders)
    if target is None:
        return None
    try:
        opened = run(["/usr/bin/open", str(target)], capture_output=True, timeout=OPEN_TIMEOUT_S).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        opened = False
    return target if opened else None



def legacy_login_item(script: Path, *, folders: list[Path] | None = None, home: Path | None = None) -> Path | None:
    """The old Sotto login item's plist, when it is ours to move or remove: it
    starts this folder's old Sotto.app, or an old Sotto.app that is gone (so
    it could only fail now)."""
    import login_item
    path = login_item.plist_path(home, login_item.LEGACY_LABEL)
    try:
        agent = plistlib.loads(path.read_bytes())
    except (OSError, plistlib.InvalidFileException, ValueError):
        return None
    if not isinstance(agent, dict) or agent.get("Label") != login_item.LEGACY_LABEL:
        return None
    arguments = agent.get("ProgramArguments")
    if not isinstance(arguments, list) or not arguments:
        return None
    for folder in folders or candidate_folders(home):
        old_app = folder / LEGACY_APP_NAME
        if arguments[0] == str(old_app / "Contents" / "MacOS" / LEGACY_EXECUTABLE):
            return path if not old_app.exists() or is_legacy_copy(old_app, script) else None
    return None


def _drop_legacy_login_item(path: Path, run) -> None:
    """Unload the old login item (launchd must never start the old app again
    this session), then delete it."""
    import login_item
    try:
        run(["/bin/launchctl", "bootout", f"gui/{os.getuid()}/{login_item.LEGACY_LABEL}"],
            capture_output=True, timeout=OPEN_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        pass
    path.unlink(missing_ok=True)


def retire_legacy(app_executable: str, script: Path, *, folders: list[Path] | None = None,
                  home: Path | None = None, run=subprocess.run) -> list[str]:
    """Scrivo.app's start: move an old Sotto login item over to Scrivo.app and
    remove the old Sotto.app, both only when this folder's installer made
    them. Returns what it did, for the log. Idempotent: a crash part way
    leaves the rest for the next start.

    Side effects: may write Scrivo's LaunchAgent plist, boot out and delete
    the old one, and delete the old app bundle."""
    import login_item
    executable = Path(app_executable.strip()) if app_executable.strip() else None
    if executable is None or executable.name != EXECUTABLE:
        return []
    folders = folders or candidate_folders(home)
    done = []
    old_login = legacy_login_item(script, folders=folders, home=home)
    if old_login is not None:
        login_item.enable(str(executable), home)  # first, so a crash never loses start-at-login
        _drop_legacy_login_item(old_login, run)
        done.append("moved the login item from Sotto.app to Scrivo.app")
    for old_app in legacy_copies(script, folders):
        shutil.rmtree(old_app)
        done.append(f"removed the old {old_app}")
    return done


def remove_legacy(script: Path, *, folders: list[Path] | None = None, home: Path | None = None,
                  run=subprocess.run) -> list[Path]:
    """Uninstall: the old Sotto login item and Sotto.app this folder's
    installer made. Returns the removed apps.

    Side effects: boots out and deletes the old login item; deletes the old app."""
    folders = folders or candidate_folders(home)
    old_login = legacy_login_item(script, folders=folders, home=home)
    if old_login is not None:
        _drop_legacy_login_item(old_login, run)
    removed = legacy_copies(script, folders)
    for old_app in removed:
        shutil.rmtree(old_app)
    return removed
