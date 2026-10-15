"""Scrivo.app builder, launcher behaviour, the login item and the move from
Sotto.app (Scrivo's earlier name) — all in temp dirs, with fakes for open and launchctl."""
import importlib.util
import os
from pathlib import Path
import plistlib
import re
import shutil
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock

import login_item
import mac_app

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("install_app", ROOT / "scripts/install_app.py")
install_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(install_app)
HAVE_CLANG = sys.platform == "darwin" and subprocess.run(
    ["/usr/bin/xcrun", "--find", "clang"], capture_output=True).returncode == 0


class LauncherSourceTests(unittest.TestCase):
    def test_paths_are_escaped_and_must_be_absolute(self):
        self.assertEqual(install_app.c_string('a "b"\\c é'), '"a \\042b\\042\\134c \\303\\251"')
        with self.assertRaises(ValueError):
            install_app.launcher_source(python=Path("python"), script=Path("/s"), data_dir=Path("/d"),
                                        hf_home=Path("/h"), app_executable=Path("/e"))

    def test_info_plist_is_a_menu_bar_app_with_a_microphone_reason(self):
        info = plistlib.loads(install_app.info_plist("0.1.0-alpha"))
        self.assertEqual(info["CFBundleIdentifier"], "io.github.zohartito.scrivo")
        self.assertEqual((info["CFBundleName"], info["CFBundleDisplayName"], info["CFBundleExecutable"],
                          info["CFBundleIconFile"]), ("Scrivo", "Scrivo", "Scrivo", "Scrivo"))
        self.assertTrue(info["NSMicrophoneUsageDescription"].startswith("Scrivo records only"))
        self.assertTrue(info["LSUIElement"])
        self.assertEqual(info["LSMinimumSystemVersion"], "14.0")
        self.assertIn("never leaves", info["NSMicrophoneUsageDescription"])


@unittest.skipUnless(HAVE_CLANG, "needs macOS with Apple's command line tools")
class LauncherBehaviourTests(unittest.TestCase):
    """Compile the real launcher around a recorder shell script (never the app)."""

    def launcher(self, body: str):
        temporary = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, temporary)
        record = temporary / "record.txt"
        script = temporary / "child.sh"
        script.write_text(f'printf "%s\\n" "$@" "$SCRIVO_DATA_DIR" "$SCRIVO_LAUNCHER" > "{record}"\n{body}\n')
        source = install_app.launcher_source(
            python=Path("/bin/sh"), script=script, data_dir=temporary / "data dir",
            hf_home=temporary / "hf", app_executable=temporary / "Scrivo")
        binary = install_app.compile_launcher(source, temporary / "Scrivo")
        return binary, record, temporary

    def test_environment_arguments_and_exit_code_pass_through(self):
        binary, record, temporary = self.launcher("exit 3")
        result = subprocess.run([str(binary)], timeout=20)
        self.assertEqual(result.returncode, 3)
        self.assertEqual(record.read_text().splitlines(),
                         ["run", "--idle-release", "0", str(temporary / "data dir"), "app"])

    def test_sigterm_is_forwarded_to_the_python_child(self):
        binary, record, _temporary = self.launcher("sleep 30")
        process = subprocess.Popen([str(binary)])
        deadline = time.monotonic() + 10
        while not record.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        process.send_signal(signal.SIGTERM)
        self.assertEqual(process.wait(timeout=10), 128 + signal.SIGTERM)

    def test_build_makes_a_signed_bundle_and_an_identical_rebuild_keeps_it(self):
        temporary = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, temporary)
        options = dict(applications=temporary / "Applications", data_dir=temporary / "data",
                       hf_home=temporary / "data" / "huggingface", python=Path(sys.executable).absolute())
        app = install_app.build(**options)
        self.assertEqual(app.name, "Scrivo.app")
        self.assertTrue((app / "Contents/MacOS/Scrivo").is_file())
        self.assertTrue((app / "Contents/Resources/Scrivo.icns").is_file())
        self.assertTrue(mac_app.launcher_runs(app, ROOT / "scrivo.py"))
        verify = subprocess.run(["/usr/bin/codesign", "--verify", str(app)], capture_output=True)
        self.assertEqual(verify.returncode, 0, verify.stderr)
        before = (app / "Contents/MacOS/Scrivo").stat().st_mtime_ns
        install_app.build(**options)
        self.assertEqual((app / "Contents/MacOS/Scrivo").stat().st_mtime_ns, before)
        # An older build of ours is replaced whole, with no staging folders left behind.
        (app / "Contents/Resources/old.txt").write_text("from an older build")
        install_app.build(**options)
        self.assertFalse((app / "Contents/Resources/old.txt").exists())
        self.assertEqual([path.name for path in options["applications"].iterdir()], ["Scrivo.app"])


class InstallLocationTests(unittest.TestCase):
    def make_app(self, folder: Path, bundle_id: str) -> Path:
        contents = folder / install_app.APP_NAME / "Contents"
        contents.mkdir(parents=True)
        (contents / "Info.plist").write_bytes(plistlib.dumps({"CFBundleIdentifier": bundle_id}))
        return contents.parent

    def test_system_applications_when_writable_otherwise_personal(self):
        home = Path("/Users/someone")
        with unittest.mock.patch.object(install_app.os, "access", return_value=True):
            self.assertEqual(install_app.default_applications(home), Path("/Applications"))
        with unittest.mock.patch.object(install_app.os, "access", return_value=False):
            self.assertEqual(install_app.default_applications(home), home / "Applications")

    def test_build_leaves_an_app_it_did_not_make_alone(self):
        with tempfile.TemporaryDirectory() as temporary:
            applications = Path(temporary)
            other = self.make_app(applications, "com.someone.else.scrivo")
            theirs = other / "Contents" / "theirs.txt"
            theirs.write_text("keep me")
            with self.assertRaises(SystemExit):
                install_app.build(applications=applications, data_dir=applications / "data",
                                  hf_home=applications / "hf", python=Path(sys.executable).absolute(),
                                  compiler="/nonexistent/clang")
            self.assertEqual(theirs.read_text(), "keep me")

    def test_only_our_bundle_is_removed_and_the_kept_copy_survives(self):
        with tempfile.TemporaryDirectory() as temporary:
            system, personal, foreign = (Path(temporary) / name for name in ("system", "personal", "foreign"))
            kept = self.make_app(system, install_app.BUNDLE_ID)
            old = self.make_app(personal, install_app.BUNDLE_ID)
            other = self.make_app(foreign, "com.someone.else.scrivo")
            removed = install_app.remove_copies([system, personal, foreign, Path(temporary) / "missing"], keep=kept)
            self.assertEqual(removed, [old])
            self.assertTrue(kept.exists())
            self.assertFalse(old.exists())
            self.assertTrue(other.exists())
            self.assertFalse(install_app.ours(other))

    def test_an_install_killed_mid_swap_keeps_the_old_app_when_the_next_copy_fails(self):
        # [F3b] Killed between the two renames, the old app survives only as
        # .Scrivo.app.outgoing (.Sotto.app.outgoing then); the next run deleted it before copying, so a
        # failed copy left no app at all.
        with tempfile.TemporaryDirectory() as temporary:
            applications = Path(temporary)
            app = applications / install_app.APP_NAME
            old = self.make_app(applications, install_app.BUNDLE_ID)
            (old / "Contents" / "old.txt").write_text("the working app")
            old.rename(applications / f".{install_app.APP_NAME}.outgoing")  # the interrupted swap
            (applications / f".{install_app.APP_NAME}.incoming").mkdir()
            staging = self.make_app(applications / "staging", install_app.BUNDLE_ID)
            with unittest.mock.patch.object(install_app.shutil, "copytree", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    install_app._swap_in(staging, app)
            self.assertEqual((app / "Contents" / "old.txt").read_text(), "the working app")

    @unittest.skipUnless(sys.platform == "darwin", "Scrivo.app is macOS only")
    def test_dialogs_use_the_icon_of_the_app_that_launched_python(self):
        import scrivo
        # Scrivo.app, and the Sotto.app of earlier versions (its launcher can still start this code).
        for app_name, name in (("Scrivo.app", "Scrivo"), ("Sotto.app", "Scrivo")):
            with self.subTest(app=app_name), tempfile.TemporaryDirectory() as temporary:
                contents = Path(temporary) / app_name / "Contents"
                (contents / "Resources").mkdir(parents=True)
                executable = contents / "MacOS" / name
                with unittest.mock.patch.dict(os.environ, {"SCRIVO_APP_EXECUTABLE": str(executable)}):
                    self.assertIsNone(scrivo.scrivo_icon_path())
                    (contents / "Resources" / f"{name}.icns").write_bytes(b"icns")
                    self.assertEqual(scrivo.scrivo_icon_path(), contents / "Resources" / f"{name}.icns")
            with unittest.mock.patch.dict(os.environ, {"SCRIVO_APP_EXECUTABLE": " "}):
                self.assertIsNone(scrivo.scrivo_icon_path())


@unittest.skipUnless(sys.platform == "darwin", "Sotto.app is macOS only")
class AccessibilityProbeTests(unittest.TestCase):
    def test_each_check_asks_a_fresh_interpreter(self):
        import scrivo
        results = iter([1, 0])
        with unittest.mock.patch("subprocess.run", side_effect=lambda argv, **kw: unittest.mock.Mock(
                returncode=next(results))) as run:
            self.assertFalse(scrivo.accessibility_granted_now())
            self.assertTrue(scrivo.accessibility_granted_now())
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args.args[0][:4], [sys.executable, "-I", "-S", "-c"])
        with unittest.mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired("python", 5)):
            self.assertFalse(scrivo.accessibility_granted_now())

    def test_probe_runs_and_answers_with_an_exit_code(self):
        import scrivo
        result = subprocess.run([sys.executable, "-I", "-S", "-c", scrivo.ACCESSIBILITY_PROBE])
        self.assertIn(result.returncode, (0, 1))


@unittest.skipUnless(sys.platform == "darwin", "macOS LaunchAgent login item; Windows: tests/test_win_startup.py")
class LoginItemTests(unittest.TestCase):
    def test_enable_writes_a_relaunch_on_crash_agent_and_disable_only_unloads_on_uninstall(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            path = login_item.enable("/Applications/Scrivo.app/Contents/MacOS/Scrivo", home)
            self.assertEqual(path, home / "Library/LaunchAgents/io.github.zohartito.scrivo.plist")
            agent = plistlib.loads(path.read_bytes())
            self.assertEqual(agent["Label"], "io.github.zohartito.scrivo")
            self.assertEqual(agent["KeepAlive"], {"SuccessfulExit": False})
            self.assertTrue(agent["RunAtLoad"])
            self.assertTrue(login_item.enabled(home))
            calls = []
            login_item.disable(home, run=lambda *a, **k: calls.append(a[0]))
            self.assertEqual(calls, [])
            self.assertFalse(login_item.enabled(home))
            login_item.enable("/x/Scrivo", home)
            login_item.disable(home, unload=True, run=lambda *a, **k: calls.append(a[0]))
            self.assertEqual(calls, [["/bin/launchctl", "bootout", f"gui/{os.getuid()}/io.github.zohartito.scrivo"]])
        with self.assertRaises(ValueError):
            login_item.agent("relative/Scrivo")


@unittest.skipUnless(sys.platform == "darwin", "macOS bash installer; Windows: tests/test_win_installer.py")
class InstallerScriptTests(unittest.TestCase):
    def test_shell_installer_parses_and_documents_its_options(self):
        script = ROOT / "scripts/install-mac.sh"
        self.assertEqual(subprocess.run(["/bin/bash", "-n", str(script)]).returncode, 0)
        self.assertTrue(os.access(script, os.X_OK))
        usage = subprocess.run(["/bin/bash", str(script), "--help"], capture_output=True, text=True)
        self.assertIn("--uninstall", usage.stdout)


FAKE_PYTHON = """#!/bin/bash
# Stands in for Python 3.12 and the venv: answers the installer's checks and
# keeps the installed packages in $FAKE_STATE (one name==version per line).
# The update's install adds newdep and then, with PIP_FAILS=1, fails; with
# PIP_HANGS=1 it hangs until stopped. PIP_CHECK_FAILS=1: pip check fails while
# newdep is installed.
case "$1 $2" in
    "-c import platform"*) echo "${FAKE_ARCH:-arm64}"; exit 0 ;;
    "-c import sys"*) echo 3.12; exit 0 ;;
esac
[ "$1 $2" = "-m pip" ] || exit 0  # scrivo.py setup, install_app.py
shift 2
pin() {  # set one name==version line in the state
    grep -v "^${1%%==*}==" "$FAKE_STATE" > "$FAKE_STATE.new"; echo "$1" >> "$FAKE_STATE.new"
    mv "$FAKE_STATE.new" "$FAKE_STATE"
}
case "$1" in
    freeze) cat "$FAKE_STATE"; exit 0 ;;
    check)
        if [ "${PIP_CHECK_FAILS:-0}" = 1 ] && grep -q '^newdep==' "$FAKE_STATE"; then
            echo "newdep 1.0 has requirement numpy<2, but you have numpy 2.0."; exit 1
        fi
        exit 0 ;;
    uninstall)
        for name in "$@"; do
            case "$name" in uninstall|-*) continue ;; esac
            echo "uninstalled: $name" >> "$FAKE_LOG"
            grep -v "^$name==" "$FAKE_STATE" > "$FAKE_STATE.new"; mv "$FAKE_STATE.new" "$FAKE_STATE"
        done
        exit 0 ;;
    install) ;;
    *) exit 0 ;;
esac
file=""
while [ $# -gt 0 ]; do [ "$1" = -r ] && file="$2"; shift; done
[ -n "$file" ] || exit 0  # pip==26.2.1
if [ "$(basename "$file")" = previous-packages.txt ]; then
    sed 's/^/restored: /' "$file" >> "$FAKE_LOG"
    while read -r line; do pin "$line"; done < "$file"
    exit 0
fi
cat "$file" "$(dirname "$file")/constraints-alpha.txt" >> "$FAKE_LOG"
pin newdep==1.0
[ "${PIP_FAILS:-0}" = 1 ] && exit 1
if [ "${PIP_HANGS:-0}" = 1 ]; then touch "$FAKE_LOG.hanging"; sleep 60; fi
while read -r line; do pin "$line"; done < "$(dirname "$file")/constraints-alpha.txt"
exit 0
"""


@unittest.skipUnless(HAVE_CLANG and shutil.which("git"), "macOS installer with git and Apple's command line tools")
class UpdateOrderTests(unittest.TestCase):
    """--update installs the new version's packages before it switches the source."""

    def git(self, cwd, *args):
        return subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                               "-c", "init.defaultBranch=main", *args],
                              cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

    def setUp(self):
        base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, base)
        remote, publisher, self.user = base / "remote.git", base / "publisher", base / "user"
        self.git(base, "init", "--bare", str(remote))
        self.git(base, "clone", str(remote), str(publisher))
        (publisher / "scripts").mkdir()
        shutil.copy2(ROOT / "scripts/install-mac.sh", publisher / "scripts/install-mac.sh")
        (publisher / "requirements-alpha.txt").write_text("-r requirements.txt\n-c constraints-alpha.txt\n")
        (publisher / "requirements.txt").write_text("numpy\n")
        (publisher / "constraints-alpha.txt").write_text("numpy==1.0\n")
        (publisher / ".gitignore").write_text("venv-alpha/\n")
        self.git(publisher, "add", ".")
        self.git(publisher, "commit", "-m", "First")
        self.git(publisher, "push", "origin", "HEAD:main")
        self.git(base, "clone", "--branch", "main", str(remote), str(self.user))
        self.old = self.git(self.user, "rev-parse", "HEAD")
        (publisher / "constraints-alpha.txt").write_text("numpy==2.0\n")
        self.git(publisher, "commit", "-am", "Newer numpy")
        self.git(publisher, "push", "origin", "HEAD:main")
        self.new = self.git(publisher, "rev-parse", "HEAD")
        fake = self.user / "venv-alpha" / "bin" / "python"
        fake.parent.mkdir(parents=True)
        fake.write_text(FAKE_PYTHON)
        fake.chmod(0o755)
        self.log = base / "pip.log"
        self.state = base / "installed.txt"
        self.state.write_text("numpy==1.0\n")
        self.env = dict(os.environ, FAKE_LOG=str(self.log), FAKE_STATE=str(self.state),
                        SCRIVO_DATA_DIR=str(base / "data"))

    def update(self, **env):
        return subprocess.run(["/bin/bash", str(self.user / "scripts/install-mac.sh"), "--update",
                               "--python", str(self.user / "venv-alpha/bin/python")],
                              env=dict(self.env, **env), capture_output=True, text=True)

    def assert_rolled_back(self, result):
        """The update failed, said so, and left the old source on exactly the old packages."""
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source was not switched", result.stderr)
        self.assertIn("previous packages are back", result.stderr)
        self.assertNotIn("Done.", result.stdout)
        self.assertIn("restored: numpy==1.0", self.log.read_text())
        self.assertEqual(self.state.read_text().split(), ["numpy==1.0"])
        self.assertEqual(self.git(self.user, "rev-parse", "HEAD"), self.old)

    def test_failed_package_install_restores_the_packages_and_leaves_the_source_alone(self):
        # pip cannot roll back a half-finished install, so the installer puts the
        # previous package set back before it reports failure.
        self.assert_rolled_back(self.update(PIP_FAILS="1"))

    def test_a_failed_pip_check_restores_the_packages(self):
        # [F2b] Only a failed pip install was rolled back; a failed pip check
        # (set -e) left the update's packages under the old source.
        result = self.update(PIP_CHECK_FAILS="1")
        self.assert_rolled_back(result)
        self.assertIn("uninstalled: newdep", self.log.read_text())  # [N11]

    def test_a_failed_source_switch_restores_the_packages(self):
        # [F2b] A failed merge --ff-only left the new packages under the old source.
        # A local edit to a file the update changes makes the fast-forward fail.
        (self.user / "constraints-alpha.txt").write_text("numpy==1.5\n")
        result = self.update()
        self.assert_rolled_back(result)
        self.assertEqual((self.user / "constraints-alpha.txt").read_text(), "numpy==1.5\n")

    def test_a_copy_with_its_own_commits_changes_nothing(self):
        # A diverged copy can never fast-forward, so it is refused before any package moves.
        (self.user / "notes.txt").write_text("mine\n")
        self.git(self.user, "add", "notes.txt")
        self.git(self.user, "commit", "-m", "My own change")
        mine = self.git(self.user, "rev-parse", "HEAD")
        result = self.update()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("its own commits", result.stderr)
        self.assertFalse(self.log.exists())  # no package install ran
        self.assertEqual(self.git(self.user, "rev-parse", "HEAD"), mine)

    def test_a_rollback_removes_the_packages_the_update_added(self):
        # [N11] The rollback reinstalled the old freeze but kept what the update added.
        self.update(PIP_FAILS="1")
        self.assertIn("uninstalled: newdep", self.log.read_text())
        self.assertNotIn("newdep", self.state.read_text())

    def test_an_update_stopped_while_installing_packages_restores_them(self):
        # [N25] Check for Updates stops a stuck update after its timeout; the
        # installer then put nothing back, leaving the update's packages under
        # the old source.
        process = subprocess.Popen(["/bin/bash", str(self.user / "scripts/install-mac.sh"), "--update",
                                    "--python", str(self.user / "venv-alpha/bin/python")],
                                   env=dict(self.env, PIP_HANGS="1"), stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            deadline = time.monotonic() + 60
            while not Path(f"{self.log}.hanging").exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            os.killpg(process.pid, signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=60)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        self.assert_rolled_back(subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr))

    def test_an_update_refuses_while_scrivo_runs_from_this_copy(self):
        # [N26] A re-run swapped source and packages under the running app.
        # The app runs the resolved path (install_app.py ROOT); a terminal run
        # (venv-alpha/bin/python scrivo.py run) a path relative to its working
        # directory, which the check missed [PR21 review].
        other = self.user.parent / "another-copy"
        other.mkdir()
        elsewhere = subprocess.Popen(["/bin/bash", "-c", "sleep 60; true", "scrivo.py", "run"], cwd=other)
        self.addCleanup(elsewhere.wait)
        self.addCleanup(elsewhere.kill)
        # sotto.py: the stand-in that launchers built before the rename still run.
        for arguments, cwd in (([str(self.user.resolve() / "scrivo.py")], None), (["scrivo.py", "run"], self.user),
                               (["./scrivo.py", "run"], self.user), (["../scrivo.py"], self.user / "scripts"),
                               ([str(self.user.resolve() / "sotto.py")], None), (["sotto.py", "run"], self.user)):
            with self.subTest(arguments=arguments, cwd=cwd):
                running = subprocess.Popen(["/bin/bash", "-c", "sleep 60; true", *arguments], cwd=cwd)
                try:
                    time.sleep(0.2)
                    refused = self.update()
                finally:
                    running.kill()
                    running.wait()
                self.assertNotEqual(refused.returncode, 0)
                self.assertIn(f"Scrivo is running from this copy (process {running.pid})", refused.stderr)
                self.assertFalse(self.log.exists())  # no package install ran
                self.assertEqual(self.git(self.user, "rev-parse", "HEAD"), self.old)
        # Check for Updates runs the installer from inside that Scrivo, which then restarts.
        # A scrivo.py in another folder (still running) is not this copy. A Scrivo
        # started before the rename names itself in SOTTO_UPDATE_FROM_PID.
        for script, variable in (("sotto.py", "SOTTO_UPDATE_FROM_PID"), ("scrivo.py", "SCRIVO_UPDATE_FROM_PID")):
            with self.subTest(variable=variable):
                running = subprocess.Popen(["/bin/bash", "-c", "sleep 60; true", script, "run"], cwd=self.user)
                self.addCleanup(running.wait)
                self.addCleanup(running.kill)
                time.sleep(0.2)
                from_app = self.update(**{variable: str(running.pid)})
                self.assertEqual(from_app.returncode, 0, from_app.stderr)
                self.assertEqual(self.git(self.user, "rev-parse", "HEAD"), self.new)
                running.kill()

    @unittest.skipUnless(any(Path(p).is_file() for p in ("/opt/homebrew/bin/python3.12", "/usr/local/bin/python3.12")),
                         "needs a native Python 3.12 in /opt/homebrew or /usr/local")
    def test_an_intel_python_first_on_path_is_skipped_for_a_native_one(self):
        # [N30] Discovery took the first python3.12 on PATH and failed if it was x86_64.
        bin_dir = self.user.parent / "bin"
        bin_dir.mkdir()
        intel = bin_dir / "python3.12"
        intel.write_text(FAKE_PYTHON)
        intel.chmod(0o755)
        result = subprocess.run(["/bin/bash", str(self.user / "scripts/install-mac.sh"), "--update"],
                                env=dict(self.env, FAKE_ARCH="x86_64", PATH=f"{bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin"),
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.git(self.user, "rev-parse", "HEAD"), self.new)
        explicit = subprocess.run(["/bin/bash", str(self.user / "scripts/install-mac.sh"), "--python", str(intel)],
                                  env=dict(self.env, FAKE_ARCH="x86_64"), capture_output=True, text=True)
        self.assertIn("is not a native arm64 Python", explicit.stderr)  # a named Python is never swapped

    def test_a_copy_that_is_not_a_git_clone_refuses_to_update(self):
        copy = self.user.parent / "archive-copy"
        shutil.copytree(self.user, copy, ignore=shutil.ignore_patterns(".git"))
        result = subprocess.run(["/bin/bash", str(copy / "scripts/install-mac.sh"), "--update",
                                 "--python", str(copy / "venv-alpha/bin/python")],
                                env=self.env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not a git clone", result.stderr)
        self.assertNotIn("Done.", result.stdout)
        self.assertFalse(self.log.exists())  # no package install ran

    def test_new_packages_are_installed_then_the_source_moves(self):
        result = self.update()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("numpy==2.0", self.log.read_text())
        self.assertEqual(self.git(self.user, "rev-parse", "HEAD"), self.new)


LEGACY_COMMIT = "21ce68b"  # the last commit before the rename: what Update Now starts from
HAVE_LEGACY_COMMIT = (ROOT / ".git").exists() and shutil.which("git") is not None and subprocess.run(
    ["git", "-C", str(ROOT), "cat-file", "-e", f"{LEGACY_COMMIT}^{{commit}}"], capture_output=True).returncode == 0


def fake_bundle(folder: Path, app_name: str, bundle_id: str, executable: str, script: Path) -> Path:
    """An app bundle whose stand-in launcher holds the script path as one C
    string among others, the way the compiled launcher does."""
    contents = folder / app_name / "Contents"
    (contents / "MacOS").mkdir(parents=True)
    (contents / "Info.plist").write_bytes(plistlib.dumps({"CFBundleIdentifier": bundle_id,
                                                          "CFBundleExecutable": executable}))
    (contents / "MacOS" / executable).write_bytes(
        b"\xcf\xfa\xed\xfe code\0SCRIVO_DATA_DIR\0" + os.fsencode(str(script)) + b"\0run\0--idle-release\0")
    return contents.parent


@unittest.skipUnless(sys.platform == "darwin", "Scrivo.app and its login item are macOS only")
class SottoToScrivoTests(unittest.TestCase):
    """Scrivo was called Sotto. After Update Now on the last Sotto build, the
    old process restarts through the old Sotto.app (or its login item), whose
    launcher runs this folder's sotto.py, now the stand-in that runs the new
    scrivo.py; that start opens Scrivo.app, and
    Scrivo.app's start retires the old app and login item. Fakes stand in for
    open and launchctl; nothing outside the temp folder is touched."""

    def setUp(self):
        base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, base)
        self.home = base / "home"
        self.folders = [base / "Applications", self.home / "Applications"]
        self.script = base / "scrivo" / "scrivo.py"
        self.other_script = base / "another-copy" / "scrivo.py"
        self.calls = []

    def fake_run(self, returncode=0, error=None):
        def run(argv, **_kwargs):
            self.calls.append(argv)
            if error is not None:
                raise error
            return subprocess.CompletedProcess(argv, returncode)
        return run

    def old_app(self, folder=None, bundle_id=mac_app.LEGACY_BUNDLE_ID, script=None) -> Path:
        # Its launcher runs the sotto.py of the folder it was built for.
        return fake_bundle(folder or self.folders[0], "Sotto.app", bundle_id, "Sotto",
                           (script or self.script).with_name(mac_app.LEGACY_SCRIPT))

    def new_app(self, folder=None, bundle_id=mac_app.BUNDLE_ID, script=None) -> Path:
        return fake_bundle(folder or self.folders[0], "Scrivo.app", bundle_id, "Scrivo", script or self.script)

    def old_login_item(self, executable: Path) -> Path:
        path = login_item.plist_path(self.home, login_item.LEGACY_LABEL)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(plistlib.dumps({"Label": "org.sotto.alpha", "ProgramArguments": [str(executable)],
                                         "RunAtLoad": True, "KeepAlive": {"SuccessfulExit": False}}))
        return path

    def retire(self, executable: Path | str):
        return mac_app.retire_legacy(str(executable), self.script, folders=self.folders, home=self.home,
                                     run=self.fake_run())

    def test_identities(self):
        self.assertEqual((mac_app.APP_NAME, mac_app.EXECUTABLE, mac_app.BUNDLE_ID),
                         ("Scrivo.app", "Scrivo", "io.github.zohartito.scrivo"))
        self.assertEqual((mac_app.LEGACY_APP_NAME, mac_app.LEGACY_BUNDLE_ID, login_item.LEGACY_LABEL),
                         ("Sotto.app", "org.sotto.alpha", "org.sotto.alpha"))
        self.assertEqual(login_item.LABEL, mac_app.BUNDLE_ID)
        self.assertEqual((install_app.APP_NAME, install_app.BUNDLE_ID), (mac_app.APP_NAME, mac_app.BUNDLE_ID))

    @unittest.skipUnless(HAVE_CLANG, "needs macOS with Apple's command line tools")
    def test_the_launcher_check_finds_the_script_path_in_the_compiled_launcher(self):
        script = self.folders[0].parent / "x" / "scrivo" / "scrivo.py"
        app = self.folders[0] / "Sotto.app"
        (app / "Contents" / "MacOS").mkdir(parents=True)
        (app / "Contents" / "Info.plist").write_bytes(plistlib.dumps({"CFBundleExecutable": "Sotto"}))
        install_app.compile_launcher(install_app.launcher_source(
            python=Path("/usr/bin/python3"), script=script, data_dir=Path("/d"), hf_home=Path("/d/hf"),
            app_executable=app / "Contents" / "MacOS" / "Sotto"), app / "Contents" / "MacOS" / "Sotto")
        self.assertTrue(mac_app.launcher_runs(app, script))
        self.assertFalse(mac_app.launcher_runs(app, self.other_script))
        self.assertFalse(mac_app.launcher_runs(app, Path("/scrivo/scrivo.py")))  # only a suffix of the real path

    def test_the_old_app_started_after_the_update_opens_scrivo_and_exits(self):
        old, new = self.old_app(), self.new_app()
        opened = mac_app.hand_off(str(old / "Contents/MacOS/Sotto"), self.script, folders=self.folders,
                                  run=self.fake_run())
        self.assertEqual(opened, new)
        self.assertEqual(self.calls, [["/usr/bin/open", str(new)]])
        self.assertTrue(old.exists(), "the hand-off itself never removes the app it runs from")

    def test_scrivo_in_the_personal_applications_folder_is_found_too(self):
        old, new = self.old_app(), self.new_app(self.folders[1])
        self.assertEqual(mac_app.hand_off(str(old / "Contents/MacOS/Sotto"), self.script, folders=self.folders,
                                          run=self.fake_run()), new)

    def test_the_old_app_keeps_running_when_there_is_no_scrivo_to_open(self):
        # Never leave the user with nothing: no Scrivo.app (its build failed),
        # one for another folder or another product, or open failing.
        old = self.old_app()
        executable = str(old / "Contents/MacOS/Sotto")
        self.assertIsNone(mac_app.hand_off(executable, self.script, folders=self.folders, run=self.fake_run()))
        self.assertEqual(self.calls, [])
        for make in (lambda: self.new_app(script=self.other_script),
                     lambda: self.new_app(bundle_id="com.someone.else.scrivo")):
            with self.subTest():
                new = make()
                self.assertIsNone(mac_app.hand_off(executable, self.script, folders=self.folders,
                                                   run=self.fake_run()))
                shutil.rmtree(new)
        self.assertEqual(self.calls, [])
        self.new_app()
        for run in (self.fake_run(returncode=1), self.fake_run(error=OSError("no open")),
                    self.fake_run(error=subprocess.TimeoutExpired("open", 30))):
            self.assertIsNone(mac_app.hand_off(executable, self.script, folders=self.folders, run=run))

    def test_a_scrivo_app_built_before_the_module_rename_is_opened_too(self):
        # Its launcher runs this folder's sotto.py (the stand-in) until the
        # update rebuilds it; a hand-off must still find it.
        old, new = self.old_app(), self.new_app(script=self.script.with_name("sotto.py"))
        self.assertEqual(mac_app.hand_off(str(old / "Contents/MacOS/Sotto"), self.script, folders=self.folders,
                                          run=self.fake_run()), new)

    def test_the_old_app_of_an_earlier_archive_beside_this_one_is_retired(self):
        # A source archive extracts to scrivo-alpha/; the one before the
        # rename extracted to sotto-alpha/ next to it, and its Sotto.app runs
        # that folder's sotto.py.
        base = self.script.parent.parent
        self.script = base / "scrivo-alpha" / "scrivo.py"
        earlier = base / "sotto-alpha" / "scrivo.py"
        self.assertEqual(mac_app.legacy_scripts(self.script),
                         [base / "scrivo-alpha" / "sotto.py", base / "sotto-alpha" / "sotto.py"])
        old, new = self.old_app(script=earlier), self.new_app()
        legacy_plist = self.old_login_item(old / "Contents/MacOS/Sotto")
        self.assertEqual(len(self.retire(new / "Contents/MacOS/Scrivo")), 2)
        self.assertFalse(old.exists())
        self.assertFalse(legacy_plist.exists())
        self.assertEqual(plistlib.loads(login_item.plist_path(self.home).read_bytes())["ProgramArguments"],
                         [str(new / "Contents/MacOS/Scrivo")])

    def test_only_an_archive_folder_takes_over_its_sibling(self):
        base = self.script.parent.parent
        self.assertEqual(mac_app.legacy_scripts(self.script), [self.script.with_name("sotto.py")])
        old = self.old_app(script=base / "sotto-alpha" / "scrivo.py")
        self.old_login_item(old / "Contents/MacOS/Sotto")
        self.assertEqual(self.retire(self.new_app() / "Contents/MacOS/Scrivo"), [])
        self.assertTrue(old.exists())

    def test_only_a_start_from_the_old_app_hands_off(self):
        new = self.new_app()
        for executable in ("", " ", str(new / "Contents/MacOS/Scrivo"), "/somewhere/Sotto",
                           str(self.folders[0] / "Other.app/Contents/MacOS/Sotto")):
            with self.subTest(executable=executable):
                self.assertIsNone(mac_app.hand_off(executable, self.script, folders=self.folders,
                                                   run=self.fake_run()))
        self.assertEqual(self.calls, [])

    def test_scrivo_moves_the_old_login_item_over_and_removes_the_old_app(self):
        old, new = self.old_app(), self.new_app()
        legacy_plist = self.old_login_item(old / "Contents/MacOS/Sotto")
        done = self.retire(new / "Contents/MacOS/Scrivo")
        self.assertEqual(len(done), 2, done)
        agent = plistlib.loads(login_item.plist_path(self.home).read_bytes())
        self.assertEqual(agent["Label"], "io.github.zohartito.scrivo")
        self.assertEqual(agent["ProgramArguments"], [str(new / "Contents/MacOS/Scrivo")])
        self.assertFalse(legacy_plist.exists())
        self.assertEqual(self.calls, [["/bin/launchctl", "bootout", f"gui/{os.getuid()}/org.sotto.alpha"]])
        self.assertFalse(old.exists())
        self.assertTrue(new.exists())
        self.calls.clear()
        self.assertEqual(self.retire(new / "Contents/MacOS/Scrivo"), [], "a second start has nothing to do")
        self.assertEqual(self.calls, [])

    def test_without_the_old_login_item_none_is_added(self):
        old, new = self.old_app(), self.new_app()
        self.assertEqual(self.retire(new / "Contents/MacOS/Scrivo"), [f"removed the old {old}"])
        self.assertFalse(login_item.enabled(self.home))
        self.assertEqual(self.calls, [])

    def test_a_login_item_whose_old_app_is_already_gone_moves_to_scrivo(self):
        new = self.new_app()
        legacy_plist = self.old_login_item(self.folders[1] / "Sotto.app/Contents/MacOS/Sotto")
        self.retire(new / "Contents/MacOS/Scrivo")
        self.assertTrue(login_item.enabled(self.home))
        self.assertFalse(legacy_plist.exists())

    def test_another_products_or_another_folders_sotto_is_left_alone(self):
        foreign = self.old_app(bundle_id="com.someone.else.sotto")
        elsewhere = self.old_app(self.folders[1], script=self.other_script)
        new = self.new_app()
        legacy_plist = self.old_login_item(elsewhere / "Contents/MacOS/Sotto")
        self.assertEqual(self.retire(new / "Contents/MacOS/Scrivo"), [])
        self.assertTrue(foreign.exists())
        self.assertTrue(elsewhere.exists())
        self.assertTrue(legacy_plist.exists())
        self.assertFalse(login_item.enabled(self.home))
        self.assertEqual(self.calls, [])
        # A login item that names something other than an old Sotto.app is not ours either.
        self.old_login_item(Path("/Applications/Something.app/Contents/MacOS/Sotto"))
        self.assertEqual(self.retire(new / "Contents/MacOS/Scrivo"), [])

    def test_a_start_that_is_not_scrivo_app_retires_nothing(self):
        old = self.old_app()
        self.old_login_item(old / "Contents/MacOS/Sotto")
        for executable in ("", str(old / "Contents/MacOS/Sotto")):  # a terminal run; the old app's fallback run
            with self.subTest(executable=executable):
                self.assertEqual(self.retire(executable), [])
        self.assertTrue(old.exists())
        self.assertEqual(self.calls, [])

    def test_an_unreadable_old_login_item_is_left_alone(self):
        old, new = self.old_app(), self.new_app()
        legacy_plist = login_item.plist_path(self.home, login_item.LEGACY_LABEL)
        legacy_plist.parent.mkdir(parents=True)
        for content in (b"not a plist", plistlib.dumps(["a list"]), plistlib.dumps({"Label": "org.sotto.alpha"}),
                        plistlib.dumps({"Label": "org.sotto.alpha", "ProgramArguments": "a string"})):
            with self.subTest(content=content[:20]):
                legacy_plist.write_bytes(content)
                self.assertIsNone(mac_app.legacy_login_item(self.script, folders=self.folders, home=self.home))
        self.assertEqual(self.retire(new / "Contents/MacOS/Scrivo"), [f"removed the old {old}"])
        self.assertTrue(legacy_plist.exists())

    def test_uninstall_removes_this_folders_old_app_and_login_item_only(self):
        old = self.old_app()
        foreign = self.old_app(self.folders[1], bundle_id="com.someone.else.sotto")
        legacy_plist = self.old_login_item(old / "Contents/MacOS/Sotto")
        removed = mac_app.remove_legacy(self.script, folders=self.folders, home=self.home, run=self.fake_run())
        self.assertEqual(removed, [old])
        self.assertFalse(legacy_plist.exists())
        self.assertFalse(login_item.enabled(self.home), "an uninstall never adds Scrivo's login item")
        self.assertTrue(foreign.exists())
        self.assertEqual(self.calls, [["/bin/launchctl", "bootout", f"gui/{os.getuid()}/org.sotto.alpha"]])

    def test_the_new_installer_keeps_the_old_app_for_the_restart_in_progress(self):
        # Update Now: the OLD install-mac.sh runs the NEW install_app.py with
        # only --data-dir, while the old Sotto still runs; its restart then
        # goes through the old Sotto.app, so that app must still be there.
        old = self.old_app()
        new = self.new_app()
        argv = ["install_app.py", "--data-dir", str(self.home / "data")]
        with unittest.mock.patch.object(sys, "argv", argv), \
                unittest.mock.patch.object(install_app, "build", return_value=new) as build, \
                unittest.mock.patch.object(install_app, "default_applications", return_value=self.folders[0]), \
                unittest.mock.patch.object(install_app, "candidate_folders", return_value=self.folders):
            install_app.main()
        self.assertEqual(build.call_args.kwargs["data_dir"], self.home / "data")
        self.assertTrue(old.exists())
        self.assertTrue(new.exists())

    def test_uninstall_removes_scrivo_and_this_folders_old_app(self):
        old, new = self.old_app(), self.new_app()
        with unittest.mock.patch.object(login_item, "disable") as disable, \
                unittest.mock.patch.object(mac_app, "remove_legacy", return_value=[old]) as remove_legacy:
            self.assertTrue(install_app.uninstall(self.folders))
        disable.assert_called_once_with(unload=True)
        remove_legacy.assert_called_once_with(install_app.ROOT / "scrivo.py", folders=self.folders)
        self.assertFalse(new.exists())

    @unittest.skipUnless(sys.platform == "darwin", "Scrivo.app is macOS only")
    def test_scrivo_py_exits_cleanly_after_handing_off_and_otherwise_retires_the_old_app(self):
        import scrivo
        retired = []
        with unittest.mock.patch.object(scrivo, "log"):
            self.assertTrue(scrivo.move_from_legacy_app(hand_off=lambda executable, script: Path("/A/Scrivo.app"),
                                                       retire=lambda executable, script: retired.append(1) or []))
            self.assertEqual(retired, [], "a hand-off never retires: Scrivo.app does that")
            self.assertFalse(scrivo.move_from_legacy_app(hand_off=lambda executable, script: None,
                                                        retire=lambda executable, script: retired.append(script) or []))
            self.assertEqual(retired, [Path(scrivo.__file__).resolve()])

            def broken(executable, script):
                raise PermissionError("Operation not permitted")
            self.assertFalse(scrivo.move_from_legacy_app(hand_off=lambda executable, script: None, retire=broken))

    @unittest.skipUnless(sys.platform == "darwin", "Scrivo.app is macOS only")
    def test_a_handed_off_start_never_reaches_the_app_or_its_permission_checks(self):
        import scrivo
        with unittest.mock.patch.object(sys, "argv", ["scrivo.py", "run", "--idle-release", "0"]), \
                unittest.mock.patch.dict(os.environ, {"SCRIVO_LAUNCHER": "app"}), \
                unittest.mock.patch.object(scrivo, "route_app_logs"), \
                unittest.mock.patch.object(scrivo, "move_from_legacy_app", return_value=True) as moved, \
                unittest.mock.patch.object(scrivo, "run") as run:
            scrivo.main()
        moved.assert_called_once_with()
        run.assert_not_called()


LAST_SCRIVO_COMMIT = "f221967"  # main before the module rename (sotto.py -> scrivo.py): the next Update Now
HAVE_LAST_SCRIVO_COMMIT = HAVE_LEGACY_COMMIT and subprocess.run(
    ["git", "-C", str(ROOT), "cat-file", "-e", f"{LAST_SCRIVO_COMMIT}^{{commit}}"], capture_output=True).returncode == 0


@unittest.skipUnless(HAVE_LAST_SCRIVO_COMMIT and HAVE_CLANG,
                     f"needs this repository's git history ({LEGACY_COMMIT}, {LAST_SCRIVO_COMMIT}) on macOS")
class UpdateFromLastSottoTests(unittest.TestCase):
    """The real first update: the installer of the last Sotto build (21ce68b)
    or of the last build before the module rename (f221967), read by bash
    while it fast-forwards its own file, finishes with this version's files.
    It runs this folder's sotto.py setup (now the stand-in that runs
    scrivo.py) with only the old SOTTO_* variables, then install_app.py."""

    def git(self, cwd, *args):
        return subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                               "-c", "init.defaultBranch=main", *args],
                              cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

    def update_from(self, commit: str, *, stand_in: bool = True):
        base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, base)
        remote, publisher, user = base / "remote.git", base / "publisher", base / "user"
        self.git(base, "init", "--bare", str(remote))
        self.git(base, "clone", str(remote), str(publisher))
        (publisher / "scripts").mkdir()
        for name in ("scripts/install-mac.sh", "scripts/install_app.py", "sotto.py"):
            (publisher / name).write_text(self.git(ROOT, "show", f"{commit}:{name}") + "\n")
        (publisher / "requirements-alpha.txt").write_text("-r requirements.txt\n-c constraints-alpha.txt\n")
        (publisher / "requirements.txt").write_text("numpy\n")
        (publisher / "constraints-alpha.txt").write_text("numpy==1.0\n")
        (publisher / ".gitignore").write_text("venv-alpha/\n")
        self.git(publisher, "add", ".")
        self.git(publisher, "commit", "-m", "The installed version")
        self.git(publisher, "push", "origin", "HEAD:main")
        self.git(base, "clone", "--branch", "main", str(remote), str(user))
        for name in ("scripts/install-mac.sh", "scripts/install_app.py", "scrivo.py", "sotto.py", "mac_app.py",
                     "scrivo_paths.py"):
            shutil.copy2(ROOT / name, publisher / name)
        if not stand_in:
            (publisher / "sotto.py").unlink()
        self.git(publisher, "add", "-A", ".")
        self.git(publisher, "commit", "-m", "Scrivo, internals renamed")
        self.git(publisher, "push", "origin", "HEAD:main")
        fake = user / "venv-alpha" / "bin" / "python"
        fake.parent.mkdir(parents=True)
        log = base / "ran.txt"
        # Record what the installer runs after the source switch, with which
        # variables; like python, a missing script file fails with status 2.
        fake.write_text(FAKE_PYTHON.replace(
            '[ "$1 $2" = "-m pip" ] || exit 0',
            'case "$1" in */install_app.py|*/sotto.py|*/scrivo.py)\n'
            '    [ -f "$1" ] || { echo "python: can\'t open file \'$1\'" >&2; exit 2; }\n'
            '    echo "ran: $* old=${SOTTO_DATA_DIR:-} new=${SCRIVO_DATA_DIR:-}" >> "$FAKE_RUN_LOG"; exit 0 ;; esac\n'
            '[ "$1 $2" = "-m pip" ] || exit 0', 1))
        fake.chmod(0o755)
        state = base / "installed.txt"
        state.write_text("numpy==1.0\n")
        data = base / "data"
        environment = {name: value for name, value in os.environ.items() if not name.startswith(("SOTTO_", "SCRIVO_"))}
        result = subprocess.run(
            ["/bin/bash", str(user / "scripts/install-mac.sh"), "--update", "--python", str(fake)],
            env=dict(environment, FAKE_LOG=str(base / "pip.log"), FAKE_STATE=str(state), FAKE_RUN_LOG=str(log),
                     SOTTO_DATA_DIR=str(data)),  # what the running app's launcher exports
            capture_output=True, text=True, timeout=120)
        return result, (log.read_text().splitlines() if log.exists() else []), user.resolve(), data

    def test_the_old_installer_runs_the_stand_in_and_the_new_install_app(self):
        for commit, closing in ((LEGACY_COMMIT, "open -a Sotto"), (LAST_SCRIVO_COMMIT, "open -a Scrivo")):
            with self.subTest(commit=commit):
                result, ran, user, data = self.update_from(commit)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue((user / "scrivo.py").is_file(), "the source moved to this version")
                self.assertIn('with_name("scrivo.py")', (user / "sotto.py").read_text(), "sotto.py is the stand-in")
                self.assertEqual(len(ran), 2, ran)
                # Only the old variable: bash went on reading the old file (git
                # replaced it, never rewrote it in place), so scrivo.py must
                # read SOTTO_DATA_DIR too.
                self.assertEqual(ran[0], f"ran: {user}/sotto.py setup old={data} new=")
                self.assertEqual(ran[1], f"ran: {user}/scripts/install_app.py --data-dir {data} old={data} new=")
                self.assertIn(closing, result.stdout)
                self.assertIn('script=ROOT / "scrivo.py"', (user / "scripts/install_app.py").read_text(),
                              "the app it rebuilds runs scrivo.py")

    def test_without_the_stand_in_the_old_installer_stops_after_the_source_moved(self):
        # Why sotto.py stays: the old installer calls it by path after the
        # merge, under set -e, and an old launcher would crash-loop on it.
        result, ran, user, _data = self.update_from(LAST_SCRIVO_COMMIT, stand_in=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("can't open file", result.stderr)
        self.assertTrue((user / "scrivo.py").is_file(), "the source had already moved")
        self.assertEqual(ran, [])


@unittest.skipUnless(sys.platform != "win32" and shutil.which("bash") and shutil.which("perl"), "needs bash and perl")
class UninstallFallbackLauncherTests(unittest.TestCase):
    """install-mac.sh --uninstall without its venv removes an old Sotto.app
    only when the launcher starts exactly this folder's scrivo.py (Codex review
    on #40: a substring match would also take a launcher built for a path that
    merely contains this one)."""

    def launcher_runs_this(self, root: Path, launcher_bytes: bytes) -> int:
        text = (ROOT / "scripts" / "install-mac.sh").read_text(encoding="utf-8")
        function = re.search(r"^launcher_runs_this\(\) \{\n.*?^\}\n", text, re.S | re.M)
        self.assertIsNotNone(function, "install-mac.sh lost launcher_runs_this")
        with tempfile.TemporaryDirectory() as temporary:
            launcher = Path(temporary) / "Scrivo"
            launcher.write_bytes(launcher_bytes)
            script = (f"ROOT={shlex.quote(str(root))}\n{function.group(0)}"
                      f"launcher_runs_this {shlex.quote(str(launcher))}\n")
            return subprocess.run(["bash", "-c", script], capture_output=True, timeout=30).returncode

    def test_only_the_exact_nul_terminated_path_counts(self):
        root = Path("/Users/someone/sotto")
        exact = os.fsencode(str(root / "sotto.py"))  # what an old Sotto.app's launcher runs
        self.assertEqual(self.launcher_runs_this(root, b"\xcf\xfa\0" + exact + b"\0rest"), 0)
        self.assertNotEqual(self.launcher_runs_this(root, b"\0/Volumes/backup" + exact + b"\0"), 0)
        self.assertNotEqual(self.launcher_runs_this(root, b"\0" + exact + b".bak\0"), 0)
        self.assertNotEqual(self.launcher_runs_this(root, b"no path here"), 0)


if __name__ == "__main__":
    unittest.main()
