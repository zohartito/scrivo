"""Scrivo was called Sotto, and its modules were sotto*.py. What keeps the
installs made before the internal rename working until their next update:
the sotto.py / sotto_win.py stand-ins, the SOTTO_* fallbacks, and the launcher
a Scrivo.app built before the rename still runs. Everything runs in temp dirs
around a recorder script, never the app."""
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import scrivo_paths

ROOT = Path(__file__).resolve().parents[1]
LAST_SCRIVO_COMMIT = "f221967"  # main before the module rename: what installed copies run now
HAVE_GIT_HISTORY = (ROOT / ".git").exists() and shutil.which("git") is not None and subprocess.run(
    ["git", "-C", str(ROOT), "cat-file", "-e", f"{LAST_SCRIVO_COMMIT}^{{commit}}"], capture_output=True).returncode == 0
HAVE_CLANG = sys.platform == "darwin" and subprocess.run(
    ["/usr/bin/xcrun", "--find", "clang"], capture_output=True).returncode == 0

# Stands in for scrivo.py / scrivo_win.py: records how it was started (with
# the real scrivo_paths beside it), then exits with EXIT_CODE.
RECORDER = '''import json, os, sys, time
from pathlib import Path
import scrivo_paths
Path(os.environ["RECORD"]).write_text(json.dumps({
    "argv": sys.argv, "orig_argv": sys.orig_argv, "file": __file__, "name": __name__, "pid": os.getpid(),
    "ppid": os.getppid(), "data_dir": str(scrivo_paths.DATA_DIR), "hf_home": str(scrivo_paths.MODEL_CACHE_DIR),
    "launcher": scrivo_paths.env("LAUNCHER"), "app_executable": scrivo_paths.env("APP_EXECUTABLE")}))
if os.environ.get("WAIT_FOR_SIGNAL"):
    Path(os.environ["RECORD"] + ".ready").write_text("")
    time.sleep(60)
raise SystemExit(int(os.environ.get("EXIT_CODE", "0")))
'''


def clean_environment(**extra) -> dict:
    """os.environ without any SCRIVO_* / SOTTO_* variable, plus extra."""
    environment = {name: value for name, value in os.environ.items() if not name.startswith(("SCRIVO_", "SOTTO_"))}
    environment.update(extra)
    return environment


class Folder:
    """A copy of the stand-ins and scrivo_paths, with recorders as the new entry points."""

    def __init__(self, test: unittest.TestCase, *, stand_ins: bool = True):
        self.path = Path(tempfile.mkdtemp()).resolve()
        test.addCleanup(shutil.rmtree, self.path)
        if stand_ins:
            for name in ("sotto.py", "sotto_win.py"):
                shutil.copy2(ROOT / name, self.path / name)
        shutil.copy2(ROOT / "scrivo_paths.py", self.path / "scrivo_paths.py")
        for name in ("scrivo.py", "scrivo_win.py"):
            (self.path / name).write_text(RECORDER, encoding="utf-8")
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


class StandInTests(unittest.TestCase):
    @unittest.skipIf(sys.platform == "win32", "the Mac stand-in execs; Windows has no exec")
    def test_sotto_py_becomes_scrivo_py_in_the_same_process(self):
        folder = Folder(self)
        data = folder.path / "data"
        process = subprocess.Popen([sys.executable, "-u", str(folder.path / "sotto.py"), "run", "--idle-release", "0"],
                                   env=clean_environment(RECORD=str(folder.record), EXIT_CODE="7",
                                                         SOTTO_DATA_DIR=str(data)))
        self.assertEqual(process.wait(timeout=60), 7, "the exit code passes through")
        record = folder.recorded()
        self.assertEqual(record["pid"], process.pid, "same process: the launcher waits on this pid")
        self.assertEqual(record["argv"], [str(folder.path / "scrivo.py"), "run", "--idle-release", "0"])
        self.assertEqual(record["orig_argv"][1:3], ["-u", str(folder.path / "scrivo.py")], "interpreter options kept")
        self.assertEqual((record["file"], record["name"]), (str(folder.path / "scrivo.py"), "__main__"))
        self.assertEqual(record["data_dir"], str(data), "the environment passes through")

    @unittest.skipIf(sys.platform == "win32", "the Mac stand-in execs; Windows has no exec")
    def test_a_signal_sent_to_the_old_script_reaches_scrivo_py(self):
        folder = Folder(self)
        process = subprocess.Popen([sys.executable, str(folder.path / "sotto.py"), "run"],
                                   env=clean_environment(RECORD=str(folder.record), WAIT_FOR_SIGNAL="1"))
        self.addCleanup(process.kill)
        deadline = time.monotonic() + 30
        while not Path(f"{folder.record}.ready").exists():
            self.assertLess(time.monotonic(), deadline, "scrivo.py never started")
            time.sleep(0.05)
        process.send_signal(signal.SIGTERM)
        self.assertEqual(process.wait(timeout=30), -signal.SIGTERM)

    def test_sotto_win_py_runs_scrivo_win_py_as_main_in_this_process(self):
        folder = Folder(self)
        process = subprocess.Popen([sys.executable, str(folder.path / "sotto_win.py"), "--tray"],
                                   env=clean_environment(RECORD=str(folder.record), EXIT_CODE="3"))
        self.assertEqual(process.wait(timeout=60), 3)
        record = folder.recorded()
        self.assertEqual(record["pid"], process.pid)
        # A tray restart re-runs sys.argv[0], so it moves to the new file.
        self.assertEqual(record["argv"], [str(folder.path / "scrivo_win.py"), "--tray"])
        self.assertEqual(record["name"], "__main__")
        self.assertEqual(Path(record["file"]).name, "scrivo_win.py")

    def test_importing_a_stand_in_runs_nothing(self):
        folder = Folder(self)
        for module in ("sotto", "sotto_win"):
            with self.subTest(module=module):
                result = subprocess.run([sys.executable, "-c", f"import {module}"], cwd=folder.path, timeout=60,
                                        env=clean_environment(RECORD=str(folder.record)), capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(folder.record.exists())

    def test_the_stand_ins_say_why_they_exist_and_when_to_delete_them(self):
        for name in ("sotto.py", "sotto_win.py"):
            with self.subTest(name=name):
                lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
                self.assertTrue(lines[0].startswith("# Stand-in: Scrivo was called Sotto"), lines[0])
                self.assertTrue(lines[1].startswith("# Delete after installs from before"), lines[1])


@unittest.skipUnless(HAVE_CLANG and HAVE_GIT_HISTORY,
                     f"needs macOS with Apple's command line tools and this repository's history "
                     f"({LAST_SCRIVO_COMMIT})")
class LauncherTests(unittest.TestCase):
    """The compiled launcher of a Scrivo.app built before the rename, and the
    one the installer builds now, around the recorder (never the app)."""

    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("install_app", ROOT / "scripts" / "install_app.py")
        self.install_app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.install_app)

    def old_launcher_source(self, **paths) -> str:
        old = subprocess.run(["git", "-C", str(ROOT), "show", f"{LAST_SCRIVO_COMMIT}:scripts/install_app.py"],
                             check=True, capture_output=True, text=True).stdout
        template = re.search(r'^LAUNCHER_C = r"""(.*?)"""$', old, re.S | re.M).group(1)
        self.assertIn('setenv("SOTTO_DATA_DIR"', template, "the launcher installed copies have")
        return template % {name: self.install_app.c_string(str(value)) for name, value in paths.items()}

    def build(self, folder: Folder, source: str) -> Path:
        binary = folder.path / "Scrivo.app" / "Contents" / "MacOS" / "Scrivo"
        binary.parent.mkdir(parents=True)
        return self.install_app.compile_launcher(source, binary)

    def test_an_old_launcher_runs_the_new_code_through_the_stand_in_with_its_old_environment(self):
        folder = Folder(self)
        data = folder.path / "data"
        executable = folder.path / "Scrivo.app" / "Contents" / "MacOS" / "Scrivo"
        launcher = self.build(folder, self.old_launcher_source(
            python=Path(sys.executable), script=folder.path / "sotto.py", data_dir=data, hf_home=data / "huggingface",
            app_executable=executable))
        process = subprocess.Popen([str(launcher)], env=clean_environment(RECORD=str(folder.record), EXIT_CODE="5"))
        self.assertEqual(process.wait(timeout=60), 5, "the launcher returns scrivo.py's exit code")
        record = folder.recorded()
        self.assertEqual(record["argv"], [str(folder.path / "scrivo.py"), "run", "--idle-release", "0"])
        self.assertEqual(record["ppid"], process.pid, "the relaunch helper waits on this parent")
        self.assertEqual((record["data_dir"], record["hf_home"]), (str(data), str(data / "huggingface")))
        self.assertEqual((record["launcher"], record["app_executable"]), ("app", str(executable)))

    def test_without_the_stand_in_an_old_launcher_fails_at_every_start(self):
        # Its login item restarts a failed start (KeepAlive SuccessfulExit=false),
        # so this would loop every 30 s with nothing on screen.
        folder = Folder(self, stand_ins=False)
        data = folder.path / "data"
        launcher = self.build(folder, self.old_launcher_source(
            python=Path(sys.executable), script=folder.path / "sotto.py", data_dir=data, hf_home=data / "huggingface",
            app_executable=folder.path / "Scrivo.app" / "Contents" / "MacOS" / "Scrivo"))
        result = subprocess.run([str(launcher)], env=clean_environment(RECORD=str(folder.record)),
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("can't open file", result.stderr)

    def test_the_rebuilt_launcher_runs_scrivo_py_with_the_new_names(self):
        folder = Folder(self, stand_ins=False)
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


@unittest.skipUnless(sys.platform == "darwin" and HAVE_GIT_HISTORY,
                     f"Scrivo.app is macOS only; needs this repository's history ({LAST_SCRIVO_COMMIT})")
class RestartAfterUpdateTests(unittest.TestCase):
    """Update Now from f221967: the old process (old code in memory) restarts
    the app by its login item's label, else by its bundle path. Both must name
    what the new installer leaves behind."""

    def old_module(self, name: str):
        source = subprocess.run(["git", "-C", str(ROOT), "show", f"{LAST_SCRIVO_COMMIT}:{name}.py"],
                                check=True, capture_output=True, text=True).stdout
        namespace = {"__name__": f"old_{name}"}
        exec(compile(source, f"{LAST_SCRIVO_COMMIT}:{name}.py", "exec"), namespace)  # this repository's own history
        return namespace

    def test_the_old_restart_and_login_item_find_the_rebuilt_app(self):
        import login_item
        import mac_app
        old_login = self.old_module("login_item")
        self.assertEqual(old_login["LABEL"], login_item.LABEL, "kickstart starts the same job")
        executable = str(Path("/Applications") / mac_app.APP_NAME / "Contents" / "MacOS" / mac_app.EXECUTABLE)
        self.assertEqual(old_login["agent"](executable), login_item.agent(executable),
                         "the login item written before the rename starts the rebuilt app unchanged")
        old_mac_app = self.old_module("mac_app")
        self.assertEqual((old_mac_app["APP_NAME"], old_mac_app["EXECUTABLE"], old_mac_app["BUNDLE_ID"]),
                         (mac_app.APP_NAME, mac_app.EXECUTABLE, mac_app.BUNDLE_ID),
                         "the bundle is rebuilt at the same path, under the same identity")


if __name__ == "__main__":
    unittest.main()
