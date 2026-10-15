#!/usr/bin/env python3
"""Run the suite with temporary Scrivo state/cache, never the caller's data."""
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]


def unsafe_ancestor(path: Path):
    """The comparator-spool guards reject any base below a group/world-writable
    or foreign-owned directory (e.g. /tmp), and many tests put temporary bases
    inside the checkout. Name the offending directory instead of failing later."""
    uid = os.getuid()
    for directory in (*reversed(path.parents), path):
        info = os.lstat(directory)
        if stat.S_ISLNK(info.st_mode) or info.st_uid not in {0, uid} or info.st_mode & 0o022:
            return directory
    return None


if os.name == 'posix' and (blocked := unsafe_ancestor(root)) is not None:
    raise SystemExit(f'Run the tests from a checkout you own outside shared folders: {blocked} '
                     'is group/world-writable or not owned by you (e.g. /tmp). Move the checkout.')

with tempfile.TemporaryDirectory(prefix='scrivo-alpha-tests-') as temporary:
    state = Path(temporary)
    # Without the old SOTTO_* names (Scrivo was called Sotto), which scrivo_paths also reads.
    inherited = {name: value for name, value in os.environ.items() if not name.startswith('SOTTO_')}
    env = dict(inherited, SCRIVO_DATA_DIR=str(state/'data'), SCRIVO_HF_HOME=str(state/'huggingface'),
               HF_HOME=str(state/'huggingface'), HF_HUB_CACHE=str(state/'huggingface/hub'),
               XDG_CACHE_HOME=str(state/'cache'), NUMBA_CACHE_DIR=str(state/'numba'),
               SCRIVO_OFFLINE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', *sys.argv[1:]],
                            cwd=root, env=env)
    raise SystemExit(result.returncode)
