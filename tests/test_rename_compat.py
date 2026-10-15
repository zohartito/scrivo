"""Scrivo was called Sotto, and its modules were sotto*.py. What still reads
the old names: the SOTTO_* fallbacks that launchers built before the internal
rename export, and the names stored outside the source; and the launcher the
installer builds now. Everything runs in temp dirs around a recorder script,
never the app."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import scrivo_paths

ROOT = Path(__file__).resolve().parents[1]
HAVE_CLANG = sys.platform == "darwin" and subprocess.run(
    ["/usr/bin/xcrun", "--find", "clang"], capture_output=True).returncode == 0

# Stands in for scrivo.py: records how it was started (with the real
# scrivo_paths beside it).
RECORDER = '''import json, os, sys
from pathlib import Path
import scrivo_paths
Path(os.environ["RECORD"]).write_text(json.dumps({
    "argv": sys.argv, "orig_argv": sys.orig_argv, "file": __file__, "name": __name__, "pid": os.getpid(),
    "ppid": os.getppid(), "data_dir": str(scrivo_paths.DATA_DIR), "hf_home": str(scrivo_paths.MODEL_CACHE_DIR),
    "launcher": scrivo_paths.env("LAUNCHER"), "app_executable": scrivo_paths.env("APP_EXECUTABLE")}))
'''


def clean_environment(**extra) -> dict:
    """os.environ without any SCRIVO_* / SOTTO_* variable, plus extra."""
    environment = {name: value for name, value in os.environ.items() if not name.startswith(("SCRIVO_", "SOTTO_"))}
    environment.update(extra)
    return environment


class Folder:
    """A copy of scrivo_paths, with a recorder as scrivo.py."""

    def __init__(self, test: unittest.TestCase):
        self.path = Path(tempfile.mkdtemp()).resolve()
        test.addCleanup(shutil.rmtree, self.path)
        shutil.copy2(ROOT / "scrivo_paths.py", self.path / "scrivo_paths.py")
        (self.path / "scrivo.py").write_text(RECORDER, encoding="utf-8")
        self.record = self.path / "record.json"

    def recorded(self) -> dict:
        return json.loads(self.record.read_text(encoding="utf-8"))


class EnvironmentFallbackTests(unittest.TestCase):
    def test_the_new_name_wins_then_the_old_one_and_blank_is_unset(self):
        self.assertEqual(scrivo_paths.env("DATA_DIR", {"SCRIVO_DATA_DIR": "/new", "SOTTO_DATA_DIR": "/old"}), "/new")
        self.assertEqual(scrivo_paths.env("DATA_DIR", {"SOTTO_DATA_DIR": " /old "}), "/old")
        self.assertEqual(scrivo_paths.env("DATA_DIR", {"SCRIVO_DATA_DIR": " ", "SOTTO_DATA_DIR": "/old"}), "/old")
        self.assertEqual(scrivo_paths.env("DATA_DIR", {"SCRIVO_DATA_DIR": "", "SOTTO_DATA_DIR": " "}), "")

    def test_an_old_launchers_data_folder_and_app_mode_are_kept(self):
        # A launcher built before the rename exports only SOTTO_*: History,
        # settings and models must stay where they are, and app mode on.
        folder = Folder(self)
        data = folder.path / "data"
        program = ("import json, scrivo_paths, lifecycle, offline_runtime; "
                   "print(json.dumps([str(scrivo_paths.DATA_DIR), str(scrivo_paths.MODEL_CACHE_DIR), "
                   "lifecycle.app_mode(), lifecycle.app_executable(), offline_runtime.offline_requested()]))")
        result = subprocess.run([sys.executable, "-c", program], cwd=ROOT, capture_output=True, text=True, timeout=60,
                                env=clean_environment(SOTTO_DATA_DIR=str(data), SOTTO_HF_HOME=str(data / "hf"),
                                                      SOTTO_LAUNCHER="app", SOTTO_APP_EXECUTABLE="/A/Scrivo",
                                                      SOTTO_OFFLINE="1"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [str(data), str(data / "hf"), True, "/A/Scrivo", True])

    def test_new_processes_and_launchers_export_only_the_new_names(self):
        import offline_runtime
        self.assertEqual(set(offline_runtime.offline_subprocess_env()) - {"PATH"}, set(offline_runtime.OFFLINE_FLAGS))
        self.assertNotIn("SOTTO_", (ROOT / "scripts" / "install_app.py").read_text(encoding="utf-8"))
        launch = (ROOT / "win_launch.py").read_text(encoding="utf-8")
        self.assertIn('os.environ["SCRIVO_DATA_DIR"]', launch)
        self.assertNotIn("SOTTO_", launch)


class KeptNamesTests(unittest.TestCase):
    def test_names_stored_outside_the_source_keep_scrivos_former_name(self):
        # Renaming these would need a migration: existing login entries,
        # one-app-per-session across old and new builds, and the data folders.
        self.assertIn('VALUE_NAME = "Sotto"', (ROOT / "win_startup.py").read_text(encoding="utf-8"))
        self.assertIn('"Local\\\\sotto-"', (ROOT / "scrivo_win.py").read_text(encoding="utf-8"))
        self.assertIn('root / ".sotto.lock"', (ROOT / "storage_lock.py").read_text(encoding="utf-8"))
        self.assertIn('"Library/Application Support/sotto"', (ROOT / "scrivo_paths.py").read_text(encoding="utf-8"))


@unittest.skipUnless(HAVE_CLANG, "needs macOS with Apple's command line tools")
class LauncherTests(unittest.TestCase):
    """The launcher the installer builds now, around the recorder (never the app)."""

    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("install_app", ROOT / "scripts" / "install_app.py")
        self.install_app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.install_app)

    def build(self, folder: Folder, source: str) -> Path:
        binary = folder.path / "Scrivo.app" / "Contents" / "MacOS" / "Scrivo"
        binary.parent.mkdir(parents=True)
        return self.install_app.compile_launcher(source, binary)

    def test_the_rebuilt_launcher_runs_scrivo_py_with_the_new_names(self):
        folder = Folder(self)
        data = folder.path / "data"
        executable = folder.path / "Scrivo.app" / "Contents" / "MacOS" / "Scrivo"
        source = self.install_app.launcher_source(
            python=Path(sys.executable), script=folder.path / "scrivo.py", data_dir=data,
            hf_home=data / "huggingface", app_executable=executable)
        self.assertNotIn("SOTTO_", source)
        launcher = self.build(folder, source)
        result = subprocess.run([str(launcher)], env=clean_environment(RECORD=str(folder.record)), timeout=60)
        self.assertEqual(result.returncode, 0)
        record = folder.recorded()
        self.assertEqual(record["argv"], [str(folder.path / "scrivo.py"), "run", "--idle-release", "0"])
        self.assertEqual((record["data_dir"], record["launcher"], record["app_executable"]),
                         (str(data), "app", str(executable)))
        self.assertTrue(self.install_app.ROOT.joinpath("scrivo.py").is_file())
        self.assertIn('script=ROOT / "scrivo.py"', (ROOT / "scripts" / "install_app.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
