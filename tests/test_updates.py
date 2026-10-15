"""Check for Updates: only on request, only a clean checkout strictly behind GitHub."""
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import updates


def completed(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


class FakeGit:
    """Answers the git commands updates.check runs, by subcommand."""

    def __init__(self, counts="0 0", status="", fetch_code=0, upstream_code=0, subjects="Fix a thing\n"):
        self.answers = {"rev-parse": completed("abc1234\n"), "fetch": completed(returncode=fetch_code,
                        stderr="fatal: unable to access github.com"),
                        "rev-list": completed(counts + "\n"), "status": completed(status), "log": completed(subjects)}
        self.upstream_code = upstream_code
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        subcommand = argv[3]
        if subcommand == "rev-parse" and "@{u}" in argv:
            return completed("origin/main\n", returncode=self.upstream_code)
        return self.answers[subcommand]


class CheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / ".git").mkdir()

    def check(self, fake, offline=False):
        return updates.check(self.root, offline=offline, run=fake, which=lambda name: "/usr/bin/git")

    def test_offline_and_non_git_copies_never_touch_the_network(self):
        fake = FakeGit()
        self.assertEqual(self.check(fake, offline=True).state, "offline")
        shutil.rmtree(self.root / ".git")
        self.assertEqual(self.check(fake).state, "not-git")
        self.assertEqual(fake.calls, [])

    def test_states_from_git(self):
        self.assertEqual(self.check(FakeGit(counts="0 0")).state, "current")
        available = self.check(FakeGit(counts="0 3", subjects="Add A\nFix B\n"))
        self.assertEqual((available.state, available.behind, available.changes), ("available", 3, ("Add A", "Fix B")))
        self.assertEqual(self.check(FakeGit(counts="1 2")).state, "diverged")
        self.assertEqual(self.check(FakeGit(counts="0 2", status=" M scrivo.py\n")).state, "local-changes")
        failed = self.check(FakeGit(fetch_code=1))
        self.assertEqual(failed.state, "failed")
        self.assertIn("unable to access", failed.detail)
        self.assertEqual(self.check(FakeGit(upstream_code=128)).state, "failed")
        self.assertEqual(updates.check(self.root, offline=False, run=FakeGit(), which=lambda name: None).state,
                         "failed")

    def test_a_failed_git_inspection_is_never_reported_as_up_to_date(self):
        broken_count = FakeGit()
        broken_count.answers["rev-list"] = completed(returncode=128, stderr="fatal: bad revision")
        self.assertEqual(self.check(broken_count).state, "failed")
        self.assertEqual(self.check(FakeGit(counts="garbage")).state, "failed")
        broken_status = FakeGit(counts="0 2")
        broken_status.answers["status"] = completed(returncode=128, stderr="fatal: index file corrupt")
        self.assertEqual(self.check(broken_status).state, "failed")

    def test_descriptions_name_the_changes_and_the_manual_path(self):
        title, text = updates.describe(updates.UpdateCheck("available", behind=7, changes=("Add A",)), "x")
        self.assertEqual(title, "7 updates available")
        self.assertIn("• Add A", text)
        self.assertIn("• …", text)
        title, text = updates.describe(updates.UpdateCheck("local-changes", "Has edits."), "run THIS")
        self.assertIn("run THIS", text)
        self.assertEqual(updates.describe(updates.UpdateCheck("current", version="abc1234"), "x")[0],
                         "Scrivo is up to date")

    def test_a_terminal_run_is_given_the_command_instead_of_an_update(self):
        # [N28] Owner decision 2026-10-09: Check for Updates in a terminal run
        # refuses and names the manual command; it used to rebuild Scrivo.app
        # for the alpha data folder, which that run may not use.
        command = updates.mac_manual_update(PurePosixPath("/Users/someone/my scrivo"))
        self.assertEqual(command, "cd '/Users/someone/my scrivo' && scripts/install-mac.sh --update")
        title, text = updates.describe(updates.UpdateCheck("available", behind=1, changes=("Add A",)),
                                       command, self_update=False)
        self.assertEqual(title, "1 update available")
        self.assertIn("• Add A", text)
        self.assertIn("started from a terminal", text)
        self.assertIn(command, text)
        self.assertNotIn("Update now?", text)
        self.assertIn("Update now?", updates.describe(updates.UpdateCheck("available", behind=1), command)[1])


@unittest.skipUnless(shutil.which("git"), "needs git")
class RealGitTests(unittest.TestCase):
    """A bare 'GitHub' remote, a publisher clone and a user clone."""

    def git(self, cwd, *args):
        return subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                               "-c", "init.defaultBranch=main", *args],
                              cwd=cwd, check=True, capture_output=True, text=True)

    def test_behind_then_local_changes_then_current(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            remote, publisher, user = base / "remote.git", base / "publisher", base / "user"
            self.git(base, "init", "--bare", str(remote))
            self.git(base, "clone", str(remote), str(publisher))
            (publisher / "README.md").write_text("one\n")
            self.git(publisher, "add", "README.md")
            self.git(publisher, "commit", "-m", "First")
            self.git(publisher, "push", "origin", "HEAD:main")
            self.git(base, "clone", "--branch", "main", str(remote), str(user))
            self.assertEqual(updates.check(user, offline=False).state, "current")

            (publisher / "README.md").write_text("two\n")
            self.git(publisher, "commit", "-am", "Faster startup")
            self.git(publisher, "push", "origin", "HEAD:main")
            result = updates.check(user, offline=False)
            self.assertEqual((result.state, result.behind, result.changes), ("available", 1, ("Faster startup",)))

            (user / "README.md").write_text("my edit\n")
            self.assertEqual(updates.check(user, offline=False).state, "local-changes")
            self.git(user, "checkout", "README.md")
            self.git(user, "pull", "--ff-only")
            self.assertEqual(updates.check(user, offline=False).state, "current")


RUNNING, ON_DISK = "c2e7f0f", "3bcf356"
STALE_NOTE = f"Scrivo's files were updated to {ON_DISK}, but the running Scrivo is still {RUNNING}."
NEW_PACKAGES = "The new version needs different packages, so a restart alone is not enough."


def stale_result(state="current", *, packages_changed=False, detail="", behind=0):
    return updates.UpdateCheck(state, detail, behind=behind, version=ON_DISK,
                               stale=updates.StaleRun(RUNNING, ON_DISK, packages_changed))


class StaleRunDialogTests(unittest.TestCase):
    """2026-10-10: a backup job's `git merge --ff-only` moved the running
    checkout from c2e7f0f to 3bcf356; Check for Updates compared the disk with
    GitHub and said "up to date" while the process still ran c2e7f0f."""

    def test_a_moved_checkout_is_never_called_up_to_date(self):
        for state in ("current", "available", "local-changes", "diverged", "offline", "failed"):
            for packages_changed in (False, True):
                for self_update in (True, False):
                    result = stale_result(state, packages_changed=packages_changed, detail="Why.",
                                          behind=2 if state == "available" else 0)
                    title, text = updates.describe(result, "MANUAL", self_update=self_update)
                    offer = updates.offer(result, self_update=self_update)
                    case = (state, packages_changed, self_update)
                    self.assertNotEqual(title, "Scrivo is up to date", case)
                    self.assertTrue(text.startswith(STALE_NOTE), case)
                    if packages_changed:  # new code on old packages: never the plain restart
                        self.assertNotEqual(offer, "restart", case)
                    if not self_update:  # [N28] a terminal run neither restarts nor updates itself
                        self.assertEqual(offer, "info", case)

    def test_the_same_packages_offer_the_restart(self):
        result = stale_result()
        self.assertEqual(updates.offer(result), "restart")
        self.assertEqual(updates.describe(result, "MANUAL"), (
            "Finish updating Scrivo",
            f"{STALE_NOTE}\n\nRestart now? A dictation in progress finishes first."))

    def test_new_packages_offer_the_installer_instead_of_a_restart(self):
        # install-mac.sh --update / update-windows.ps1 install the packages of
        # @{u}, which is HEAD when the check found nothing newer.
        result = stale_result(packages_changed=True)
        self.assertEqual(updates.offer(result), "update")
        self.assertEqual(updates.describe(result, "MANUAL"), (
            "Finish updating Scrivo",
            f"{STALE_NOTE} {NEW_PACKAGES}\n\nUpdate now? Scrivo installs them, then restarts, "
            "usually within a minute."))

    def test_new_packages_without_github_name_the_command(self):
        # Update Now would contact GitHub: not while offline, and not when it
        # just could not be reached.
        manual = "run scripts/install-mac.sh --update in Terminal from Scrivo's source folder."
        for state, detail in (("offline", "Offline mode is on, so Scrivo does not contact GitHub."),
                              ("failed", "Could not reach GitHub (no network?).")):
            result = stale_result(state, packages_changed=True, detail=detail)
            self.assertEqual(updates.offer(result), "info")
            self.assertEqual(updates.describe(result, manual), (
                "Finish updating Scrivo",
                f"{STALE_NOTE} {NEW_PACKAGES}\n\n{detail}\n\nQuit Scrivo, then run scripts/install-mac.sh "
                "--update in Terminal from Scrivo's source folder. Then start Scrivo again."))
        offline_restart = stale_result("offline", detail="Offline mode is on, so Scrivo does not contact GitHub.")
        self.assertEqual(updates.offer(offline_restart), "restart")
        self.assertEqual(updates.describe(offline_restart, manual)[1],
                         f"{STALE_NOTE}\n\nOffline mode is on, so Scrivo does not contact GitHub.\n\n"
                         "Restart now? A dictation in progress finishes first.")

    def test_a_terminal_run_is_told_what_to_do(self):
        command = updates.mac_manual_update(PurePosixPath("/Users/someone/scrivo"))
        self.assertEqual(updates.describe(stale_result(), command, self_update=False), (
            "Finish updating Scrivo",
            f"{STALE_NOTE}\n\nScrivo was started from a terminal, so it does not restart itself. "
            "Quit it, then start it again."))
        self.assertEqual(updates.describe(stale_result(packages_changed=True), command, self_update=False), (
            "Finish updating Scrivo",
            f"{STALE_NOTE} {NEW_PACKAGES}\n\nScrivo was started from a terminal, so it does not update "
            f"itself. Quit it, then run in Terminal:\n{command}\nThen start it again."))

    def test_github_ahead_too_offers_the_update_with_the_note(self):
        # The update installs @{u}'s packages before it switches the source,
        # so it covers new packages on the disk commit as well.
        result = updates.UpdateCheck("available", behind=1, changes=("Add A",), version=ON_DISK,
                                     stale=updates.StaleRun(RUNNING, ON_DISK, True))
        self.assertEqual(updates.offer(result), "update")
        plain = updates.describe(updates.UpdateCheck("available", behind=1, changes=("Add A",)), "MANUAL")
        self.assertEqual(updates.describe(result, "MANUAL"), (plain[0], f"{STALE_NOTE}\n\n{plain[1]}"))
        hand = stale_result("diverged", packages_changed=True, detail="This copy has its own commits.")
        self.assertEqual(updates.offer(hand), "info")
        self.assertEqual(updates.describe(hand, "MANUAL"), (
            "Update by hand", f"{STALE_NOTE}\n\nThis copy has its own commits.\n\nTo update anyway: MANUAL"))

    def test_a_moved_checkout_with_its_own_commits_or_edits_still_offers_the_restart(self):
        # The updater refuses such a copy, but the restart only runs the files
        # already on disk: with the same packages it is all that is needed.
        for state, detail in (("diverged", "This copy has its own commits."),
                              ("local-changes", "This copy has local changes.")):
            result = stale_result(state, detail=detail)
            self.assertEqual(updates.offer(result), "restart", state)
            self.assertEqual(updates.describe(result, "MANUAL"), (
                "Finish updating Scrivo",
                f"{STALE_NOTE}\n\n{detail}\n\nRestart now? A dictation in progress finishes first."), state)
            self.assertEqual(updates.offer(result, self_update=False), "info", state)
            self.assertEqual(updates.describe(result, "MANUAL", self_update=False)[1],
                             f"{STALE_NOTE}\n\n{detail}\n\nScrivo was started from a terminal, so it does not "
                             "restart itself. Quit it, then start it again.", state)

    def test_without_a_stale_run_the_offers_are_unchanged(self):
        for state, app, terminal in (("current", "info", "info"), ("available", "update", "info"),
                                     ("not-git", "releases", "releases"), ("failed", "error", "error"),
                                     ("offline", "info", "info"), ("local-changes", "info", "info"),
                                     ("diverged", "info", "info")):
            result = updates.UpdateCheck(state)
            self.assertEqual((updates.offer(result), updates.offer(result, self_update=False)),
                             (app, terminal), state)


@unittest.skipUnless(shutil.which("git"), "needs git")
class StartedFromRealGitTests(unittest.TestCase):
    """A bare 'GitHub' remote, a publisher clone and the folder Scrivo runs from."""

    def git(self, cwd, *args):
        return subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                               "-c", "init.defaultBranch=main", *args],
                              cwd=cwd, check=True, capture_output=True, text=True)

    def publish(self, publisher, name, text, message):
        (publisher / name).write_text(text)
        self.git(publisher, "add", name)
        self.git(publisher, "commit", "-m", message)
        self.git(publisher, "push", "origin", "HEAD:main")

    def test_a_merge_under_a_running_scrivo_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            remote, publisher, user = base / "remote.git", base / "publisher", base / "user"
            self.git(base, "init", "--bare", str(remote))
            self.git(base, "clone", str(remote), str(publisher))
            for name in updates.PACKAGE_LISTS:
                (publisher / name).write_text("pin==1\n")
            self.publish(publisher, "scrivo.py", "one\n", "First")
            self.git(base, "clone", "--branch", "main", str(remote), str(user))

            started = updates.started_from(user)  # once, at launch
            self.assertEqual(len(started), 40)
            current = updates.check(user, offline=False, started_from=started)
            self.assertEqual((current.state, current.stale), ("current", None))

            # The backup job: a code-only change merged under the running app.
            self.publish(publisher, "scrivo.py", "two\n", "Faster startup")
            self.git(user, "fetch", "--quiet")
            self.git(user, "merge", "--ff-only", "--quiet", "origin/main")
            moved = self.git(user, "rev-parse", "HEAD").stdout.strip()
            result = updates.check(user, offline=False, started_from=started)
            self.assertEqual(result.state, "current")
            self.assertEqual(result.stale, updates.StaleRun(started[:7], moved[:7], False))
            self.assertNotEqual(updates.describe(result, "x")[0], "Scrivo is up to date")
            self.assertEqual(updates.offer(result), "restart")

            # A pin moved too: the plain restart would start new code on old packages.
            self.publish(publisher, updates.PACKAGE_LISTS[-1], "pin==2\n", "Bump a pin")
            self.git(user, "pull", "--ff-only", "--quiet")
            result = updates.check(user, offline=False, started_from=started)
            self.assertTrue(result.stale.packages_changed)
            self.assertEqual(updates.offer(result), "update")
            # Local git only: still reported with offline mode on.
            offline = updates.check(user, offline=True, started_from=started)
            self.assertEqual((offline.state, offline.stale), ("offline", result.stale))

    def test_no_commit_is_recorded_outside_a_git_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(updates.started_from(Path(temporary)), "")
            result = updates.check(Path(temporary), offline=False, started_from="")
            self.assertEqual((result.state, result.stale), ("not-git", None))


class ApplyMacTests(unittest.TestCase):
    """The installer's update path, and whether it switched the source."""

    def apply(self, heads, installer_code, output):
        heads = iter(heads)
        installer_calls = []

        def git(argv, **kwargs):
            return completed(next(heads) + "\n")

        def popen(argv, **kwargs):
            installer_calls.append((argv, kwargs))
            kwargs["stdout"].write(output)
            return mock.Mock(**{"wait.return_value": installer_code})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = updates.apply_mac(root, "/venv/bin/python", root / "logs" / "update.log", run=git, popen=popen)
        self.assertEqual([argv for argv, _kwargs in installer_calls],
                         [["/bin/bash", str(root / "scripts" / "install-mac.sh"), "--update",
                           "--python", "/venv/bin/python"]])
        kwargs = installer_calls[0][1]
        self.assertTrue(kwargs["start_new_session"])  # [N25] stopped as a whole group
        self.assertEqual(kwargs["env"]["SCRIVO_UPDATE_FROM_PID"], str(os.getpid()))  # [N26]
        return result

    def test_a_finished_update_reports_its_last_line_and_the_new_source(self):
        self.assertEqual(self.apply(["aaa", "bbb"], 0, "== Scrivo.app\nDone.\n"), (True, "Done.", True))

    def test_a_failure_before_the_switch_leaves_the_source_alone(self):
        self.assertEqual(self.apply(["aaa", "aaa"], 1, "✗ Installing packages failed.\n"),
                         (False, "✗ Installing packages failed.", False))

    def test_a_failure_after_the_switch_says_the_source_changed(self):
        self.assertEqual(self.apply(["aaa", "bbb"], 1, "✗ model download failed\n"),
                         (False, "✗ model download failed", True))

    @unittest.skipIf(sys.platform == "win32", "the Mac updater")
    def test_a_stuck_update_is_stopped_with_everything_it_started(self):
        # [N25] The timeout killed bash but not its pip child, which went on
        # changing packages after Scrivo had reported the update failed.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            child = root / "child.pid"
            (root / "scripts" / "install-mac.sh").write_text(
                f"/bin/bash -c 'echo $$ > \"{child}\"; sleep 60; true'\necho finished\n")
            with mock.patch.object(updates, "UPDATE_TIMEOUT_S", 2):
                finished, tail, source_changed = updates.apply_mac(root, sys.executable,
                                                                    root / "logs" / "update.log")
            pid = int(child.read_text())
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and _alive(pid):
                    time.sleep(0.05)
                self.assertFalse(_alive(pid), "the installer's child kept running")
            finally:
                if _alive(pid):
                    os.kill(pid, signal.SIGKILL)
            self.assertFalse(finished)
            self.assertFalse(source_changed)
            self.assertIn("stopped", tail)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@unittest.skipUnless(sys.platform == "win32", "Windows tray updater")
class WindowsUpdateTests(unittest.TestCase):
    def test_git_never_opens_a_console_window(self):
        # [F37] The tray runs under pythonw; each git call used to flash a console.
        kwargs_seen = []
        fake = FakeGit(counts="0 2")

        def run(argv, **kwargs):
            kwargs_seen.append(kwargs)
            return fake(argv, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / ".git").mkdir()
            self.assertEqual(updates.check(Path(temporary), offline=False, run=run,
                                           which=lambda name: "git").state, "available")
        self.assertEqual(len(kwargs_seen), 6)
        for kwargs in kwargs_seen:
            self.assertTrue(kwargs.get("creationflags", 0) & subprocess.CREATE_NO_WINDOW, kwargs)

    def test_the_check_is_given_the_commit_scrivo_started_from(self):
        import offline_runtime
        import scrivo_win
        seen = []

        def check(root, *, offline, started_from):
            seen.append((root, offline, started_from))
            return updates.UpdateCheck("current")
        started = "c2e7f0f" + "0" * 33
        with mock.patch.object(updates, "check", check), \
                mock.patch.object(offline_runtime, "offline_requested", return_value=False):
            scrivo_win.Controller(started_from=started).check_updates()
            scrivo_win.Controller().check_updates()
        root = Path(scrivo_win.__file__).resolve().parent
        self.assertEqual(seen, [(root, False, started), (root, False, "")])

    def test_hand_off_waits_for_quiet_and_reports_once(self):
        import queue
        import scrivo
        import scrivo_win
        import win_ui
        capture = mock.Mock()
        capture.is_active.return_value = False
        engine = mock.Mock()
        engine.force_finish.return_value = False  # nothing is recording

        def lifecycle_parts():
            return dict(engine=engine, on_finish=mock.Mock(),
                        capture_gate=scrivo.CaptureGate(), lifecycle={"closed": None})

        busy_jobs = queue.Queue()
        busy_jobs.put("live")
        busy = scrivo_win.Controller(shutdown=scrivo.ShutdownBoundary(), capture=capture, jobs=busy_jobs,
                                    deliveries=queue.Queue(), finishing=lambda: False,
                                    **lifecycle_parts())
        with mock.patch("subprocess.Popen") as spawn:
            self.assertIn("Finish the current dictation", busy.update_and_restart())
        spawn.assert_not_called()
        idle = scrivo_win.Controller(shutdown=scrivo.ShutdownBoundary(), capture=capture, jobs=queue.Queue(),
                                    deliveries=queue.Queue(), finishing=lambda: False,
                                    **lifecycle_parts())
        with mock.patch("subprocess.Popen") as spawn, mock.patch.object(scrivo_win, "log"):
            idle.update_and_restart()
            idle.shutdown.event.wait(5)  # the quit drains on its own thread
        argv = spawn.call_args.args[0]
        self.assertTrue(argv[0].lower().startswith("powershell"))
        self.assertTrue(argv[argv.index("-File") + 1].endswith("update-windows.ps1"))
        self.assertTrue(idle.shutdown.requested())
        with tempfile.TemporaryDirectory() as temporary:
            status = Path(temporary) / "update-status.txt"
            status.write_text("﻿ok abc1234\n", encoding="utf-8")
            self.assertEqual(win_ui.update_report(temporary), "Scrivo updated (abc1234).")
            self.assertIsNone(win_ui.update_report(temporary))
            status.write_text("failed Scrivo did not quit, so nothing was changed.", encoding="utf-8")
            self.assertEqual(win_ui.update_report(temporary),
                             "Update failed: Scrivo did not quit, so nothing was changed.")


if __name__ == "__main__":
    unittest.main()
