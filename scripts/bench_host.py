"""Local Python host with TCC purpose strings, confined to .venv-bench.

No system Python, Terminal bundle, Scrivo bundle or privacy setting is changed.
Run this script once with .venv-bench/bin/python before using the real backends.
"""
from __future__ import annotations

import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys


def host_bundle():
    prefix = Path(sys.prefix)
    if prefix.name != ".venv-bench" or not (prefix / "pyvenv.cfg").is_file():
        raise RuntimeError("Use the isolated .venv-bench environment.")
    return prefix / "ScrivoBenchPython.app"


def setup():
    from Foundation import NSBundle

    source = NSBundle.mainBundle()
    destination = host_bundle()
    if destination.exists():
        raise RuntimeError("The benchmark host already exists; it was left unchanged.")
    executable = destination / "Contents/MacOS/Python"
    executable.parent.mkdir(parents=True)
    try:
        # Copy the actual framework interpreter, not bin/python's launcher.
        shutil.copy2(source.executablePath(), executable)
        info = {
            "CFBundleExecutable": "Python", "CFBundleIdentifier": "io.github.zohartito.scrivo.bench",
            "CFBundleName": "Scrivo Benchmark", "CFBundlePackageType": "APPL", "CFBundleVersion": "1",
            "NSSpeechRecognitionUsageDescription": "Compare private recordings using on-device speech recognition.",
            "NSMicrophoneUsageDescription": "Record a benchmark clip only when you start the recording command.",
            "LSUIElement": True,
        }
        (destination / "Contents/Info.plist").write_bytes(plistlib.dumps(info))
        subprocess.run(["codesign", "--force", "--sign", "-", str(destination)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except BaseException:
        shutil.rmtree(destination)
        raise


def ensure_host():
    """Re-exec the same CLI in the prepared host, retaining this venv's packages."""
    from Foundation import NSBundle

    if NSBundle.mainBundle().bundleIdentifier() == "io.github.zohartito.scrivo.bench":
        return
    executable = host_bundle() / "Contents/MacOS/Python"
    if not executable.is_file():
        from scripts.bench_common import BenchError

        raise BenchError("Prepare the permission host with .venv-bench/bin/python scripts/bench_host.py first.")
    environment = dict(os.environ, __PYVENV_LAUNCHER__=str(Path(sys.prefix) / "bin/python"))
    # On macOS this variable keeps sys.prefix and sys.executable inside the venv.
    os.execve(executable, [str(executable), *sys.argv], environment)


if __name__ == "__main__":
    try:
        setup()
        print("Prepared the isolated benchmark permission host. No permissions were requested.")
    except Exception:
        print("Could not prepare the benchmark host. Use .venv-bench on macOS; an existing host is left unchanged.",
              file=sys.stderr)
        raise SystemExit(1) from None
