"""Check for updates: compare this git checkout with its GitHub branch.

Sotto never contacts the network on its own. This runs only when the user
chooses Check for Updates, and not at all while an offline flag is set. It
updates only a clean checkout that is strictly behind its upstream branch, so
local changes are never overwritten.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path, PurePath
import shlex
import shutil
import signal
import subprocess
import sys

RELEASES_URL = "https://github.com/zohartito/sotto/releases"
MAX_LISTED_CHANGES = 5
UPDATE_TIMEOUT_S = 1800
# Time a stopped Mac update gets to put the previous packages back (one pip run).
UPDATE_STOP_GRACE_S = 300
# The Windows tray runs under pythonw: without this flag each git call would
# flash a console window.
_NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
# The files this platform's updater installs packages from: install-mac.sh's
# requirements-alpha.txt (with what it includes), update-windows.ps1's lists.
PACKAGE_LISTS = (("requirements-alpha-windows.txt", "requirements-alpha-windows-cuda.txt",
                  "constraints-alpha-windows.txt", "constraints-alpha-windows-cuda.txt")
                 if sys.platform == "win32" else
                 ("requirements-alpha.txt", "requirements.txt", "constraints-alpha.txt"))
SHORT_SHA = 7


@dataclass(frozen=True)
class StaleRun:
    """The folder's HEAD moved after this process started (a git pull or merge
    by hand), so the running code is older than the files on disk."""
    running: str          # short commit this process started from
    on_disk: str          # short commit checked out now
    packages_changed: bool  # PACKAGE_LISTS differ: a plain restart would run new code on old packages


@dataclass(frozen=True)
class UpdateCheck:
    # current | available | not-git | local-changes | diverged | offline | failed
    state: str
    detail: str = ""
    behind: int = 0
    changes: tuple[str, ...] = ()
    version: str = ""
    stale: StaleRun | None = None


def _git(root: Path, *args: str, timeout: float = 30, run=subprocess.run):
    return run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=timeout,
               **_NO_WINDOW)


def _last_line(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1][:200] if lines else ""


def check(root: Path, *, offline: bool, started_from: str = "", run=subprocess.run,
          which=shutil.which) -> UpdateCheck:
    """Compare this checkout with GitHub, and its HEAD with started_from (the
    commit this process started from; local git only, offline too)."""
    result = _check(root, offline=offline, run=run, which=which)
    stale = stale_run(root, started_from, run=run)
    return replace(result, stale=stale) if stale else result


def _check(root: Path, *, offline: bool, run, which) -> UpdateCheck:
    if offline:
        return UpdateCheck("offline", "Offline mode is on, so Sotto does not contact GitHub.")
    if not (root / ".git").exists():
        return UpdateCheck("not-git", "This copy was not installed with git, so it cannot update "
                                      "itself. Download the latest release from GitHub.")
    if which("git") is None:
        return UpdateCheck("failed", "Git is not installed, so Sotto cannot check for updates.")
    try:
        version = _git(root, "rev-parse", "--short", "HEAD", run=run).stdout.strip()
        if _git(root, "rev-parse", "--abbrev-ref", "@{u}", run=run).returncode != 0:
            return UpdateCheck("failed", "This copy does not follow a GitHub branch.", version=version)
        fetched = _git(root, "fetch", "--quiet", timeout=90, run=run)
        if fetched.returncode != 0:
            reason = _last_line(fetched.stderr) or "no network?"
            return UpdateCheck("failed", f"Could not reach GitHub ({reason}).", version=version)
        compared = _git(root, "rev-list", "--left-right", "--count", "HEAD...@{u}", run=run)
        counts = compared.stdout.split()
        if compared.returncode != 0 or len(counts) != 2 or not all(count.isdigit() for count in counts):
            return UpdateCheck("failed", "Could not compare this copy with GitHub.", version=version)
        ahead, behind = int(counts[0]), int(counts[1])
        if behind == 0:
            return UpdateCheck("current", version=version)
        if ahead:
            return UpdateCheck("diverged", "This copy has its own commits, so Sotto will not update it "
                                           "automatically.", behind=behind, version=version)
        status = _git(root, "status", "--porcelain", "--untracked-files=no", run=run)
        if status.returncode != 0:
            return UpdateCheck("failed", "Could not read this copy's git status.", version=version)
        if status.stdout.strip():
            return UpdateCheck("local-changes", "This copy has local changes, so Sotto will not update "
                                                "it automatically.", behind=behind, version=version)
        subjects = _git(root, "log", "--format=%s", f"-n{MAX_LISTED_CHANGES}", "HEAD..@{u}",
                        run=run).stdout.splitlines()
        return UpdateCheck("available", behind=behind, version=version,
                           changes=tuple(line.strip() for line in subjects if line.strip()))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return UpdateCheck("failed", f"Could not check for updates ({str(exc)[:120]}).")


def _head(root: Path, run) -> str:
    try:
        return _git(root, "rev-parse", "HEAD", run=run).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def started_from(root: Path, run=subprocess.run) -> str:
    """The commit checked out at root; run() reads it once at launch (local
    git only). Empty outside a git checkout or when git cannot answer."""
    return _head(root, run) if (root / ".git").exists() else ""


def stale_run(root: Path, started: str, *, package_lists=PACKAGE_LISTS, run=subprocess.run) -> StaleRun | None:
    """How the running code (started) differs from the checkout's HEAD now, or
    None when it does not or either is unknown. Local git only."""
    if not started or not (root / ".git").exists():
        return None
    now = _head(root, run)
    if not now or now == started:
        return None
    try:
        # 0: the package lists are the same; 1: they differ; anything else
        # (a commit git cannot read) counts as differing, never as the same.
        same = _git(root, "diff", "--quiet", started, now, "--", *package_lists, run=run).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        same = False
    return StaleRun(started[:SHORT_SHA], now[:SHORT_SHA], not same)


def _stop_group(installer) -> None:
    """Stop the installer and everything it started (pip included): SIGTERM
    makes install-mac.sh put the previous packages back; SIGKILL whatever is
    left after UPDATE_STOP_GRACE_S."""
    try:
        os.killpg(installer.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        installer.wait(timeout=UPDATE_STOP_GRACE_S)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(installer.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    installer.wait()


def apply_mac(root: Path, python: str, log_path: Path, run=subprocess.run,
              popen=subprocess.Popen) -> tuple[bool, str, bool]:
    """Run the installer's own update path (fetch, the new version's pinned
    packages, then the source switch, model check and Sotto.app) while Sotto
    keeps running.

    Returns (finished, last log line, source_changed). The installer switches
    the source only after the packages installed, so a failure with
    source_changed False left this copy's code untouched.

    Side effects: the installer runs in its own process group, named as
    started from this Sotto (SOTTO_UPDATE_FROM_PID, so its running-Sotto check
    lets it through); after UPDATE_TIMEOUT_S that whole group is stopped.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    before = _head(root, run)
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            installer = popen(["/bin/bash", str(root / "scripts" / "install-mac.sh"), "--update", "--python", python],
                              cwd=str(root), stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                              env=dict(os.environ, SOTTO_UPDATE_FROM_PID=str(os.getpid())))
            try:
                finished, stopped = installer.wait(timeout=UPDATE_TIMEOUT_S) == 0, False
            except subprocess.TimeoutExpired:
                _stop_group(installer)
                finished, stopped = False, True
        tail = _last_line(log_path.read_text(encoding="utf-8", errors="replace"))
        if stopped:
            tail = f"The update took over {UPDATE_TIMEOUT_S // 60} minutes and was stopped. {tail}".strip()
    except OSError as exc:
        finished, tail = False, f"The update did not finish ({str(exc)[:120]})."
    after = _head(root, run)
    return finished, tail, bool(before and after and before != after)


def mac_manual_update(root: PurePath) -> str:
    """The Terminal command that updates the checkout at root by hand."""
    return f"cd {shlex.quote(str(root))} && scripts/install-mac.sh --update"


def offer(result: UpdateCheck, *, self_update: bool = True) -> str:
    """What the Check for Updates dialog offers: "update" (Update Now, the
    installer's update path), "restart" (the drained restart), "releases",
    "error" or "info" (nothing to do but read). A stale run is never offered
    the plain restart when its packages changed, nor Update Now without a
    fresh look at GitHub; a terminal run neither restarts nor updates (N28).
    The restart only runs the files already on disk, so a copy with its own
    commits or edits (which the updater refuses) is offered it too."""
    stale = result.stale
    if result.state == "available":
        return "update" if self_update else "info"
    if stale:
        if not self_update:
            return "info"
        if not stale.packages_changed:
            return "restart"
        return "update" if result.state == "current" else "info"
    if result.state == "not-git":
        return "releases"
    if result.state == "failed":
        return "error"
    return "info"


def describe(result: UpdateCheck, manual_update: str, *, self_update: bool = True) -> tuple[str, str]:
    """Dialog title and text for a check result; manual_update is the
    platform's command for updating by hand. self_update False (a Mac run
    not started by Sotto.app) names that command instead of offering the
    update, whose installer rebuilds and restarts Sotto.app (N28). A stale
    run (result.stale) is said first, in every state."""
    stale = result.stale
    if stale is None:
        return _describe(result, manual_update, self_update=self_update)
    note = f"Sotto's files were updated to {stale.on_disk}, but the running Sotto is still {stale.running}."
    by_hand = result.state in ("local-changes", "diverged")
    if result.state == "available" or (by_hand and stale.packages_changed):
        title, text = _describe(result, manual_update, self_update=self_update)
        return title, f"{note}\n\n{text}"
    detail = f"\n\n{result.detail}" if (by_hand or result.state in ("offline", "failed")) and result.detail else ""
    if not stale.packages_changed:
        if self_update:
            action = "Restart now? A dictation in progress finishes first."
        else:
            action = "Sotto was started from a terminal, so it does not restart itself. Quit it, then start it again."
        return "Finish updating Sotto", f"{note}{detail}\n\n{action}"
    note = f"{note} The new version needs different packages, so a restart alone is not enough."
    if offer(result, self_update=self_update) == "update":
        action = "Update now? Sotto installs them, then restarts, usually within a minute."
    elif self_update:
        action = f"Quit Sotto, then {manual_update} Then start Sotto again."
    else:
        action = (f"Sotto was started from a terminal, so it does not update itself. Quit it, then run in "
                  f"Terminal:\n{manual_update}\nThen start it again.")
    return "Finish updating Sotto", f"{note}{detail}\n\n{action}"


def _describe(result: UpdateCheck, manual_update: str, *, self_update: bool) -> tuple[str, str]:
    if result.state == "current":
        return "Sotto is up to date", f"You have the latest version ({result.version})."
    if result.state == "available":
        count = f"{result.behind} update" + ("s" if result.behind != 1 else "")
        listed = "\n".join(f"• {subject}" for subject in result.changes)
        more = "\n• …" if result.behind > len(result.changes) else ""
        if not self_update:
            return f"{count} available", (f"{listed}{more}\n\nSotto was started from a terminal, so it does "
                                          f"not update itself. Quit it, then run in Terminal:\n{manual_update}")
        return f"{count} available", (f"{listed}{more}\n\nUpdate now? Sotto restarts when it is done, "
                                      "usually within a minute.")
    if result.state in ("local-changes", "diverged"):
        return "Update by hand", f"{result.detail}\n\nTo update anyway: {manual_update}"
    return "Could not check for updates", result.detail
