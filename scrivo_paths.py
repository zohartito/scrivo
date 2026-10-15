"""Local storage locations; overrides keep alpha/test state separate.

Scrivo was called Sotto. The data folders keep that name (moving them would
mean migrating History, recordings and models), and every override is read
as SCRIVO_<name>, else SOTTO_<name>: launchers, shortcuts and updaters built
before the rename export only the old names.
"""
import os
from pathlib import Path
import sys

LEGACY_ENV_PREFIX = "SOTTO_"


def env(name: str, environ=None) -> str:
    """SCRIVO_<name>, else SOTTO_<name> (the old name); blank counts as unset."""
    environ = os.environ if environ is None else environ
    return (environ.get(f"SCRIVO_{name}", "").strip()
            or environ.get(f"{LEGACY_ENV_PREFIX}{name}", "").strip())


def _override(name: str, *, renamed: bool = True):
    """A non-blank override path; a blank value counts as unset, never as cwd."""
    value = env(name) if renamed else os.environ.get(name, "").strip()
    return Path(value).expanduser().absolute() if value else None


_data_override = _override("DATA_DIR")
if sys.platform == "win32":
    # Windows keeps the pre-alpha location: history, preferences and the model
    # cache all live under %APPDATA%\sotto unless overridden.
    DATA_DIR = _data_override or (
        _override("APPDATA", renamed=False) or Path.home() / "AppData" / "Roaming") / "sotto"
    MODEL_CACHE_DIR = _override("HF_HOME") or DATA_DIR / "huggingface"
else:
    DATA_DIR = _data_override or Path.home() / "Library/Application Support/sotto"
    # Preserve the existing cache location on case-sensitive volumes too.
    MODEL_CACHE_DIR = _override("HF_HOME") or (
        DATA_DIR / "huggingface" if _data_override
        else Path.home() / "Library/Application Support/Sotto/huggingface")
