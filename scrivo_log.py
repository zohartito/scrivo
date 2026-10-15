"""Scrivo's log: lines on stderr, which Scrivo.app routes to a small rotating
file. Logs hold counts and timings only, never what was said.

Moved verbatim out of scrivo.py so the modules split from it (delivery,
lifecycle, gestures, capture) can log without importing scrivo. scrivo re-exports the
public names, so ``scrivo.log is scrivo_log.log``; the rotation state stays
private here, so patch it here (``scrivo_log._point_output_at``).
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path


def log(msg: str) -> None:
    """Never raises: the transcription worker logs from inside its own error
    handler, so a full disk or a rotation race must not end the thread (N35)."""
    try:
        print(msg, file=sys.stderr, flush=True)
        keep_app_log_small()
    except (OSError, ValueError):
        pass


_app_log: dict = {"path": None, "max_bytes": 0}  # set once logs are routed to a file
_app_log_lock = threading.Lock()  # every thread logs; one of them rotates at a time


def route_app_logs(data_dir: Path, max_bytes: int = 2_000_000) -> Path:
    """Scrivo.app has no terminal: send stdout/stderr to a small rotating log.
    Logs hold counts and timings only, never what was said."""
    folder = data_dir / "logs"
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = folder / "scrivo.log"
    _rotate_if_large(path, max_bytes)
    _point_output_at(path)
    _app_log.update(path=path, max_bytes=max_bytes)
    return path


def _rotate_if_large(path: Path, max_bytes: int) -> bool:
    """Move a log past max_bytes to scrivo.log.1 (replacing the older one)."""
    if path.exists() and path.stat().st_size > max_bytes:
        os.replace(path, path.with_suffix(".log.1"))
        return True
    return False


def _point_output_at(path: Path) -> None:
    """Send this process's stdout and stderr to the log file at path."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    for stream in (1, 2):
        os.dup2(descriptor, stream)
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
