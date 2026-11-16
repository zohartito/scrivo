"""Scrivo's log: lines on stderr, which Scrivo.app routes to a small rotating
file. Logs hold counts and timings only, never what was said.

Moved verbatim out of scrivo.py so the modules split from it (delivery,
lifecycle, gestures, capture) can log without importing scrivo. scrivo re-exports the
public names, so ``scrivo.log is scrivo_log.log``; the rotation state stays
private here, so patch it here (``scrivo_log._point_output_at``).
"""
from __future__ import annotations

import os
import stat
import sys
import threading
from contextlib import contextmanager
from pathlib import Path


def log(msg: str) -> None:
    """Never raises: the transcription worker logs from inside its own error
    handler, so a full disk or a rotation race must not end the thread (N35)."""
    try:
        print(msg, file=sys.stderr, flush=True)
        keep_app_log_small()
    except (OSError, ValueError):
        pass


def enable_stack_dump_signal() -> None:
    """``kill -USR1 <pid>`` writes every thread's Python stack to the log.

    A frozen Scrivo says nothing otherwise (2026-10-19: hotkey dead, process
    idle, nothing logged, no debugger allowed). faulthandler's handler runs in
    C, so it works with every Python thread blocked. Frames only (file, line,
    function): never text or values. Never raises."""
    try:
        import faulthandler
        import signal
        if hasattr(signal, "SIGUSR1"):
            faulthandler.register(signal.SIGUSR1, all_threads=True)
    except Exception:  # diagnostics only: never ends startup
        pass


_app_log: dict = {"path": None, "max_bytes": 0}  # set once logs are routed to a file
_app_log_lock = threading.Lock()  # every thread logs; one of them rotates at a time


def route_app_logs(data_dir: Path, max_bytes: int = 2_000_000) -> Path:
    """Scrivo.app has no terminal: send stdout/stderr to a small rotating log.
    Logs hold counts and timings only, never what was said."""
    folder = data_dir / "logs"
    path = folder / "scrivo.log"
    _rotate_if_large(path, max_bytes)
    _point_output_at(path)
    _app_log.update(path=path, max_bytes=max_bytes)
    return path


@contextmanager
def _log_directory(folder: Path):
    """Open/create the private log directory without following any symlink.

    Descriptor-relative traversal is for the macOS app. Windows uses its own
    console logger; retain this helper's portability for headless tests.
    """
    folder = Path(os.path.abspath(folder))
    if os.name != "posix":
        if any(part.is_symlink() for part in (folder, *folder.parents)):
            raise RuntimeError("unsafe log directory")
        folder.mkdir(parents=True, exist_ok=True)
        yield None
        return
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory = os.open(folder.anchor, flags)
    try:
        for index, part in enumerate(folder.parts[1:], 1):
            try:
                child = os.open(part, flags, dir_fd=directory)
            except FileNotFoundError:
                try:
                    os.mkdir(part, 0o700, dir_fd=directory)
                except FileExistsError:
                    pass
                child = os.open(part, flags, dir_fd=directory)
            try:
                info = os.fstat(child)
                if info.st_uid not in {0, os.getuid()} or info.st_mode & 0o022:
                    raise RuntimeError("unsafe log directory")
                if index == len(folder.parts) - 1:
                    if info.st_uid != os.getuid():
                        raise RuntimeError("log directory is not owned by this user")
                    os.fchmod(child, 0o700)
            except BaseException:
                os.close(child)
                raise
            os.close(directory)
            directory = child
        yield directory
    finally:
        os.close(directory)


def _open_log(path: Path, directory: int | None, *, create: bool = False) -> int:
    flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path.name if directory is not None else path,
                         flags | (os.O_CREAT if create else 0), 0o600, dir_fd=directory)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError("log is not a regular file with a single link")
        if os.name == "posix":
            if info.st_uid != os.getuid():
                raise RuntimeError("log is not owned by this user")
            os.fchmod(descriptor, 0o600)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _rotate_if_large(path: Path, max_bytes: int) -> bool:
    """Move a log past max_bytes to scrivo.log.1 (replacing the older one)."""
    with _log_directory(path.parent) as directory:
        try:
            descriptor = _open_log(path, directory)
        except FileNotFoundError:
            return False
        try:
            if os.fstat(descriptor).st_size > max_bytes:
                if directory is None:
                    os.replace(path, path.with_suffix(".log.1"))
                else:
                    os.replace(path.name, path.with_suffix(".log.1").name,
                               src_dir_fd=directory, dst_dir_fd=directory)
                return True
        finally:
            os.close(descriptor)
    return False


def _point_output_at(path: Path) -> None:
    """Send this process's stdout and stderr to the log file at path."""
    with _log_directory(path.parent) as directory:
        descriptor = _open_log(path, directory, create=True)
        try:
            for stream in (1, 2):
                os.dup2(descriptor, stream)
        finally:
            os.close(descriptor)


def keep_app_log_small() -> None:
    """Rotate a routed log that grew past its limit while Scrivo runs; the old
    file keeps the open descriptors, so point output at a fresh one."""
    path = _app_log["path"]
    if path is None:
        return
    with _app_log_lock:
        if _rotate_if_large(path, _app_log["max_bytes"]):
            _point_output_at(path)
