"""N9b: a finished capture is on disk from key-up until its History row exists.

Before N9b a recording lived only in RAM between key-up and its History row,
so a crash, a SIGKILL or a logout that skipped the drain lost it. Now the
key-up hands the audio to ``history.CaptureSpool`` (written off the event-tap
thread), the worker or teardown deletes the file once the capture's row
exists or its fate is decided, and the next launch turns any file still there
into a "[not transcribed: Scrivo stopped]" row, exactly like the N9 keep.
"""
from __future__ import annotations

import ast
import os
import queue
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import types
import unittest
import unittest.mock
import uuid
from pathlib import Path

import numpy as np

import scrivo
from audio_codec import encode_pcm16
from speech_config import resolve_speech_config

ROOT = Path(__file__).resolve().parents[1]
WHISPER = resolve_speech_config("auto", language="en")
NOT_TRANSCRIBED = "[not transcribed: Scrivo stopped]"
WAIT_S = 10.0


def speech(seconds=1.5, rate=48000, amplitude=0.1):
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def wait_for(predicate, timeout=WAIT_S):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


def spooled(base: Path) -> list[Path]:
    """Every file in the spool folder but its owner lock: audio and partial writes."""
    folder = base / "capture-spool"
    return sorted(path for path in folder.iterdir() if path.name != ".owner") if folder.exists() else []


def writer_idle(spool) -> bool:
    return spool._queue.unfinished_tasks == 0


_OPEN_SPOOLS: list = []  # closed after each test: Windows cannot delete a folder whose lock file is open


def new_spool(base: Path, logs=None):
    from history import CaptureSpool
    spool = CaptureSpool(base, log=(logs.append if logs is not None else lambda line: None))
    _OPEN_SPOOLS.append(spool)
    assert spool.claim(), "a fresh data folder's spool could not be claimed"
    return spool


def close_open_spools():
    while _OPEN_SPOOLS:
        _OPEN_SPOOLS.pop().close()


def open_history(base: Path, *, tolerate_unreadable=False):
    from history import HistoryStore
    from learning import LearningStore
    store = HistoryStore(base, tolerate_unreadable=tolerate_unreadable)
    learning = LearningStore(base, tolerate_unreadable=tolerate_unreadable)
    return store, scrivo.history_coordinator(store, learning)


def recover(spool, store, coordinator, logs):
    return scrivo.recover_capture_spool(spool, store, coordinator, model=WHISPER.model_repo,
                                       speech_config=WHISPER, glossary_terms=(), log=logs.append)


class TemporaryData(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix="scrivo-spool-")
        self.addCleanup(folder.cleanup)
        self.base = Path(folder.name) / "data"
        self.base.mkdir()
        self.addCleanup(close_open_spools)  # runs before the folder cleanup


class CaptureSpoolTests(TemporaryData):
    def test_n9b_a_finished_capture_is_on_disk_until_it_is_released(self):
        spool, raw, capture_id = new_spool(self.base), speech(1.0), uuid.uuid4().hex
        spool.submit(capture_id, raw, 48000.0, 1_760_000_000.25)
        self.assertTrue(wait_for(lambda: writer_idle(spool)), "the spool write never finished")
        files = spooled(self.base)
        self.assertEqual([path.name for path in files], [f"{capture_id}-1760000000250.wav"])
        from audio_codec import read_pcm16_wav
        _, identity = read_pcm16_wav(files[0])
        self.assertEqual(identity.sha256, encode_pcm16(raw, 48000)[1].sha256)
        if sys.platform != "win32":
            self.assertEqual(stat.S_IMODE(files[0].stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(spool.directory.stat().st_mode), 0o700)
        spool.release(capture_id)
        self.assertEqual(spooled(self.base), [])

    def test_n9b_a_release_before_the_write_leaves_nothing(self):
        spool = new_spool(self.base)
        started, proceed = threading.Event(), threading.Event()
        real_write = spool._write

        def slow_write(*args):
            started.set()
            proceed.wait(WAIT_S)
            real_write(*args)
        spool._write = slow_write
        capture_id = uuid.uuid4().hex
        spool.submit(capture_id, speech(1.0), 48000.0, time.time())
        self.assertTrue(started.wait(WAIT_S))
        spool.release(capture_id)  # its row exists before the file does
        proceed.set()
        self.assertTrue(wait_for(lambda: writer_idle(spool)))
        self.assertEqual(spooled(self.base), [], "a released capture outlived its row on disk")

    def test_n9b_nothing_is_written_while_idle(self):
        spool = new_spool(self.base)
        time.sleep(0.05)
        self.assertEqual(spooled(self.base), [])
        self.assertIsNone(spool._writer, "an idle Scrivo started the spool writer")

    def test_n9b_a_second_scrivo_on_the_same_data_folder_neither_spools_nor_recovers(self):
        from history import CaptureSpool
        first = new_spool(self.base)
        first.submit(uuid.uuid4().hex, speech(1.0), 48000.0, time.time())
        self.assertTrue(wait_for(lambda: writer_idle(first)))
        second = CaptureSpool(self.base, log=lambda line: None)
        _OPEN_SPOOLS.append(second)
        self.assertFalse(second.claim())
        second.submit(uuid.uuid4().hex, speech(1.0), 48000.0, time.time())
        self.assertEqual(second.leftovers(), [])
        self.assertEqual(len(spooled(self.base)), 1, "the second copy wrote or took a capture")

    def test_n9b_a_run_that_ends_hands_the_spool_to_the_next(self):
        # Windows restarts into a new process before the old one exits.
        from history import CaptureSpool
        first = new_spool(self.base)
        first.submit(uuid.uuid4().hex, speech(1.0), 48000.0, time.time())
        first.close()
        self.assertEqual(len(spooled(self.base)), 1, "close did not wait for the queued write")
        second = CaptureSpool(self.base, log=lambda line: None)
        _OPEN_SPOOLS.append(second)
        self.assertTrue(second.claim())
        self.assertEqual(len(second.leftovers()), 1)


class FakeClock:
    """time.monotonic and time.sleep for take_data_folder: sleeping moves the clock."""

    def __init__(self, on_sleep=None, on_monotonic=None):
        self.now, self.on_sleep, self.on_monotonic, self.slept = 1000.0, on_sleep, on_monotonic, 0.0

    def monotonic(self):
        if self.on_monotonic is not None:
            self.on_monotonic(self)
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        self.slept += seconds
        if self.on_sleep is not None:
            self.on_sleep(self)


SCRIVO_ARGS = "/opt/homebrew/bin/python3.12 -u /Users/someone/scrivo/scrivo.py run --idle-release 0"
RECORDED_AT = 1_790_000_000.0
# Started a minute before its record: the process that wrote it.
HOLDER = scrivo.ProcessState(started=RECORDED_AT - 60.0, activity=5000, suspended=False)
SUSPENDED = HOLDER._replace(suspended=True)

HOLDER_CHILD = """
import sys, time
sys.path.insert(0, sys.argv[1])
from history import CaptureSpool
spool = CaptureSpool(sys.argv[2], log=print)
assert spool.claim()
spool.record_owner()
print("ready", flush=True)
while True:
    time.sleep(0.05)
"""


def moving(states, pid):
    """on_sleep: ``pid`` makes system calls while the launch looks."""
    def tick(clock):
        if pid in states:
            states[pid] = states[pid]._replace(activity=states[pid].activity + 1)
    return tick


@unittest.skipUnless(sys.platform == "darwin", "one Scrivo per data folder is macOS-only (Windows has its named mutex)")
class DataFolderOwnerTests(TemporaryData):
    """2026-11-01: swap ran out and macOS suspended Scrivo. The suspended copy
    kept the data folder, its hotkey and its menu bar slot, ran no SIGTERM
    handler, and opening Scrivo.app again started a second copy beside it."""

    def holder(self, pid=4242, *, phase="listening"):
        """Another Scrivo holding the folder, its record written at RECORDED_AT."""
        spool = new_spool(self.base)
        self.record(pid, phase=phase)
        return spool

    def record(self, pid, written=RECORDED_AT, *, phase="listening"):
        owner = self.base / "capture-spool" / ".owner"
        owner.write_text(f"{pid:>10} {phase:<9}\n" if pid is not None else "", encoding="ascii")
        os.utime(owner, (written, written))

    def take(self, clock, states, *, args=SCRIVO_ARGS, kill=None, logs=None):
        """take_data_folder with ``states`` as the kernel's view of each pid."""
        from history import CaptureSpool
        spool = CaptureSpool(self.base, log=lambda line: None)
        _OPEN_SPOOLS.append(spool)
        self.kill = kill or unittest.mock.Mock(side_effect=AssertionError("killed a Scrivo that was not suspended"))
        outcome = scrivo.take_data_folder(
            spool, state_of=states.get, args_of=lambda pid: args, kill=self.kill, monotonic=clock.monotonic,
            sleep=clock.sleep, log=(logs if logs is not None else []).append)
        return outcome, spool

    def killing(self, holder):
        """kill() for ``holder``: the lock goes with the process."""
        return unittest.mock.Mock(side_effect=lambda pid, sig: holder.close())

    def test_the_holder_records_its_pid_and_phase_in_one_fixed_width_write(self):
        spool = new_spool(self.base)
        (spool.directory / ".owner").write_text("123456789012345 something longer\n", encoding="ascii")
        spool.record_owner()
        self.assertEqual((spool.directory / ".owner").read_bytes(), f"{os.getpid():>10} starting \n".encode())
        pid, written, listening = spool.owner()
        self.assertEqual((pid, listening), (os.getpid(), False))
        self.assertAlmostEqual(written, time.time(), delta=5)
        spool.record_owner("listening")
        self.assertEqual((spool.directory / ".owner").read_bytes(), f"{os.getpid():>10} listening\n".encode())
        self.assertEqual(spool.owner()[::2], (os.getpid(), True))
        spool.record_owner("stopping")
        self.assertEqual(spool.owner()[::2], (os.getpid(), False))

    def test_a_record_that_cannot_be_written_is_logged_not_raised(self):
        # It is written at startup and as shutdown begins: neither may die on it.
        logs = []
        spool = new_spool(self.base, logs)
        with unittest.mock.patch("os.pwrite", side_effect=OSError(30, "Read-only file system")):
            spool.record_owner("stopping")
        self.assertEqual(len(logs), 1)
        self.assertIn("could not record this Scrivo (stopping)", logs[0])

    def test_a_free_data_folder_is_taken_and_recorded(self):
        outcome, spool = self.take(FakeClock(), {})
        self.assertEqual(outcome, "owner")
        self.assertTrue(spool.claimed)
        self.assertEqual(spool.owner()[0], os.getpid())

    def test_a_running_scrivo_keeps_its_data_folder_and_this_launch_is_told_so(self):
        self.holder()
        states = {4242: HOLDER}
        clock = FakeClock(on_sleep=moving(states, 4242))
        outcome, spool = self.take(clock, states)
        self.assertEqual(outcome, "running")
        self.assertFalse(spool.claimed)
        self.assertEqual(clock.slept, scrivo.OWNER_POLL_S, "a running holder answers at the first look")

    def test_a_scrivo_macos_suspended_is_killed_and_its_data_folder_taken(self):
        holder, logs, clock = self.holder(phase="starting"), [], FakeClock()  # even one still starting
        kill = self.killing(holder)
        outcome, spool = self.take(clock, {4242: SUSPENDED}, kill=kill, logs=logs)
        self.assertEqual(outcome, "owner")
        self.assertEqual(spool.owner()[0], os.getpid())
        kill.assert_called_once_with(4242, scrivo.signal.SIGKILL)
        self.assertGreaterEqual(clock.slept, scrivo.OWNER_CONFIRM_S, "it was killed before it was watched")
        self.assertIn("pid 4242", logs[0])
        self.assertIn("suspended", logs[0])

    def test_a_listening_scrivo_that_never_moves_is_killed(self):
        # However it was frozen: once listening, its watchdog threads move every second.
        holder = self.holder(phase="listening")
        kill = self.killing(holder)
        outcome, _ = self.take(FakeClock(), {4242: HOLDER}, kill=kill)
        self.assertEqual(outcome, "owner")
        kill.assert_called_once_with(4242, scrivo.signal.SIGKILL)

    def test_a_scrivo_starting_up_or_stopping_is_left_alone_while_nothing_in_it_moves(self):
        # A model load may wait in native code with nothing else running yet,
        # however long ago it started (a Mac that slept through its warmup);
        # a teardown may wait on History's lock after its watchdogs stopped.
        for phase in ("starting", "stopping"):
            with self.subTest(phase):
                self.holder(phase=phase)
                outcome, spool = self.take(FakeClock(), {4242: HOLDER._replace(started=RECORDED_AT - 86400.0)})
                self.assertEqual(outcome, "running")
                self.assertFalse(spool.claimed)
                close_open_spools()

    def test_a_holder_that_exits_while_the_launch_watches_hands_the_folder_over(self):
        holder, states = self.holder(), {4242: HOLDER}

        def exit_holder(clock):
            states.pop(4242, None)
            holder.close()
        outcome, _ = self.take(FakeClock(on_sleep=exit_holder), states)
        self.assertEqual(outcome, "owner")

    def test_a_record_whose_pid_now_belongs_to_a_newer_process_is_not_the_holder(self):
        # An earlier holder's pid, reused by a Scrivo that started just after the record.
        self.holder()
        clock = FakeClock()
        outcome, spool = self.take(clock, {4242: SUSPENDED._replace(started=RECORDED_AT + 0.5)})
        self.assertEqual(outcome, "shared")
        self.assertFalse(spool.claimed)
        self.assertGreaterEqual(clock.slept, scrivo.OWNER_PUBLISH_WAIT_S, "the record was not read again")

    def test_a_holder_slow_to_write_its_record_is_still_found(self):
        # It took the lock; seconds later the record still names the one before it.
        self.holder(pid=4242)
        states = {5151: HOLDER._replace(started=RECORDED_AT + 100.0)}

        def publish_late(clock):
            if clock.slept == 3.0:
                self.record(5151, written=RECORDED_AT + 200.0)
            moving(states, 5151)(clock)
        outcome, _ = self.take(FakeClock(on_sleep=publish_late), states)
        self.assertEqual(outcome, "running")

    def test_a_holder_that_is_not_scrivo_or_left_no_pid_is_never_killed(self):
        cases = {"no pid (a build from before the record)": (None, SCRIVO_ARGS),
                 "a pid now used by another program": (4242, "/usr/bin/vim notes.txt"),
                 "a pid that no longer exists": (4343, SCRIVO_ARGS),
                 "this very process": (os.getpid(), SCRIVO_ARGS)}
        for label, (pid, args) in cases.items():
            with self.subTest(label):
                holder = self.holder(pid)
                outcome, spool = self.take(FakeClock(), {4242: SUSPENDED, os.getpid(): SUSPENDED}, args=args)
                self.assertEqual(outcome, "shared")
                self.assertFalse(spool.claimed)
                holder.close()

    def test_a_relative_script_path_is_still_scrivo(self):
        # venv-alpha/bin/python scrivo.py run: the documented source launch.
        self.holder()
        states = {4242: HOLDER}
        outcome, _ = self.take(FakeClock(on_sleep=moving(states, 4242)), states,
                               args="venv-alpha/bin/python scrivo.py run")
        self.assertEqual(outcome, "running")

    def test_a_holder_it_may_not_kill_leaves_the_launch_running_as_before(self):
        self.holder()
        outcome, _ = self.take(FakeClock(), {4242: SUSPENDED},
                               kill=unittest.mock.Mock(side_effect=PermissionError("not permitted")))
        self.assertEqual(outcome, "shared")

    def test_a_launch_that_loses_the_folder_does_not_listen_beside_the_winner(self):
        # launchd's KeepAlive, or another launch, takes the folder first: after
        # the kill, or when the holder exits on its own while it is watched.
        for how in ("killed", "exited"):
            with self.subTest(how):
                outcome, spool = self.lose_the_folder(how)
                self.assertEqual(outcome, "running")
                self.assertFalse(spool.claimed)
                close_open_spools()

    def lose_the_folder(self, how):
        holder, states = self.holder(), {4242: SUSPENDED if how == "killed" else HOLDER}

        def another_launch_wins(*_args):
            if 4242 not in states:
                return
            holder.close()
            states.pop(4242)
            winner = new_spool(self.base)
            self.assertTrue(winner.claimed)
            self.record(5151, written=RECORDED_AT + 7200.0)
            states[5151] = HOLDER._replace(started=RECORDED_AT + 7000.0)

        def tick(clock):
            if how == "exited" and clock.slept >= 2.0:
                another_launch_wins()
            moving(states, 5151)(clock)
        return self.take(FakeClock(on_sleep=tick), states, kill=unittest.mock.Mock(side_effect=another_launch_wins))

    def test_a_folder_freed_as_the_settle_time_runs_out_is_taken_not_refused(self):
        # The holder exits on its own; its lock goes only after the last look.
        holder, states, gone = self.holder(), {4242: HOLDER}, []

        def exit_holder(clock):
            if clock.slept >= 1.0:
                states.pop(4242, None)
                gone.append(True)

        def lock_goes(clock):
            if gone and holder.claimed:
                holder.close()
        clock = FakeClock(on_sleep=exit_holder, on_monotonic=lock_goes)
        with unittest.mock.patch.multiple(scrivo, OWNER_SETTLE_S=1.0, OWNER_KILL_WAIT_S=0.0):
            outcome, spool = self.take(clock, states)
        self.assertEqual(outcome, "owner")
        self.assertTrue(spool.claimed)

    def test_the_kernel_view_of_a_real_process(self):
        me = scrivo.process_state(os.getpid())
        self.assertLess(me.started, time.time())
        self.assertFalse(me.suspended)
        self.assertIsNone(scrivo.process_state(2 ** 22 + 12345), "a pid that cannot exist")

    def test_a_real_stopped_holder_is_replaced_and_a_real_running_one_is_not(self):
        # The kernel's view for real: libproc's flags and counters, ps, SIGKILL.
        folder = Path(tempfile.mkdtemp(prefix="scrivo-holder-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(folder, ignore_errors=True))
        script = folder / "scrivo.py"
        script.write_text(HOLDER_CHILD, encoding="utf-8")
        child = subprocess.Popen([sys.executable, str(script), str(ROOT), str(self.base)],
                                 stdout=subprocess.PIPE, text=True)
        self.addCleanup(lambda: (child.kill(), child.wait(), child.stdout.close()))
        self.assertEqual(child.stdout.readline().strip(), "ready")
        from history import CaptureSpool
        logs = []
        with unittest.mock.patch.object(scrivo, "OWNER_CONFIRM_S", 2.0):
            first = CaptureSpool(self.base, log=lambda line: None)
            _OPEN_SPOOLS.append(first)
            self.assertEqual(scrivo.take_data_folder(first, log=logs.append), "running")
            os.kill(child.pid, scrivo.signal.SIGSTOP)
            self.assertTrue(wait_for(lambda: scrivo.process_state(child.pid).suspended), "the holder never stopped")
            time.sleep(0.2)  # its last system calls on the way into the stop
            second = CaptureSpool(self.base, log=lambda line: None)
            _OPEN_SPOOLS.append(second)
            self.assertEqual(scrivo.take_data_folder(second, log=logs.append), "owner")
        self.assertEqual(child.wait(timeout=WAIT_S), -scrivo.signal.SIGKILL)
        self.assertEqual(second.owner()[0], os.getpid())
        self.assertIn(f"pid {child.pid}", logs[-1])


@unittest.skipIf(sys.platform == "win32", "Scrivo.app and launchd are macOS-only")
class SecondCopyTests(unittest.TestCase):
    def refuse(self, environ):
        with unittest.mock.patch.dict(os.environ, environ), \
                unittest.mock.patch("ui.notice_dialog") as dialog, \
                unittest.mock.patch("scrivo.log"), \
                self.assertRaises(SystemExit) as stopped:
            scrivo.refuse_second_copy()
        return stopped.exception.code, dialog.called

    def test_a_second_copy_exits_and_only_scrivo_app_opened_by_hand_says_why(self):
        import login_item
        cases = {"a terminal run": ({"SCRIVO_LAUNCHER": "", "XPC_SERVICE_NAME": "0"}, (1, False)),
                 "Scrivo.app opened again": ({"SCRIVO_LAUNCHER": "app",
                                              "XPC_SERVICE_NAME": "application.io.github.zohartito.scrivo.1.2"},
                                             (0, True)),
                 # KeepAlive restarts the login item after its frozen copy was
                 # killed: no dialog, and 0 so launchd stops there.
                 "the login item": ({"SCRIVO_LAUNCHER": "app", "XPC_SERVICE_NAME": login_item.LABEL}, (0, False)),
                 "the sealed LaunchAgent": ({"SCRIVO_LAUNCHER": "", "XPC_SERVICE_NAME": scrivo.LAUNCHD_LABEL},
                                            (0, False))}
        for label, (environ, expected) in cases.items():
            with self.subTest(label):
                self.assertEqual(self.refuse({"SOTTO_LAUNCHER": "", **environ}), expected)


class RecoveryTests(TemporaryData):
    def die_with_a_spooled_capture(self) -> tuple[str, float, np.ndarray]:
        """Spool one capture in a separate process that then exits without any
        teardown, as a crash, SIGKILL or a logout that skips the drain would."""
        capture_id, captured_ts, raw = uuid.uuid4().hex, 1_760_000_123.5, speech(1.2)
        audio = self.base.parent / "raw.npy"
        np.save(audio, raw)
        script = textwrap.dedent(f"""
            import os, sys, time
            sys.path.insert(0, {str(ROOT)!r})
            import numpy as np
            from history import CaptureSpool
            spool = CaptureSpool({str(self.base)!r}, log=print)
            assert spool.claim()
            spool.submit({capture_id!r}, np.load({str(audio)!r}), 48000.0, {captured_ts!r})
            deadline = time.monotonic() + 10
            while spool._queue.unfinished_tasks and time.monotonic() < deadline:
                time.sleep(0.01)
            os._exit(0)  # no drain, no teardown, no History row
        """)
        env = dict(os.environ, SCRIVO_DATA_DIR=str(self.base), SCRIVO_HF_HOME=str(self.base / "hf"))
        done = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True,
                              timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)
        return capture_id, captured_ts, raw

    def test_n9b_a_capture_left_by_a_process_that_died_becomes_a_row_with_its_audio(self):
        capture_id, captured_ts, raw = self.die_with_a_spooled_capture()
        self.assertEqual(len(spooled(self.base)), 1, "the dead process left no spooled capture")
        store, coordinator = open_history(self.base)
        logs: list[str] = []
        waiting = recover(new_spool(self.base), store, coordinator, logs)
        self.assertEqual(waiting, 0)
        rows = store.entries(10)
        self.assertEqual([row["text"] for row in rows], [NOT_TRANSCRIBED], logs)
        row = rows[0]
        self.assertEqual(row["ts"], captured_ts)
        self.assertEqual(row["attempts"][-1]["provenance"], "live_suspect")
        self.assertEqual(row["preprocessing"]["outcome"], "suspect")
        self.assertEqual(row["audio"]["raw"]["sha256"], encode_pcm16(raw, 48000)[1].sha256)
        snapshot = coordinator.snapshot_for_retry(row["id"])  # Retry works
        self.assertEqual(len(snapshot.samples), round(len(raw) * 16000 / 48000))
        self.assertEqual(spooled(self.base), [], "a recovered capture stayed in the spool")
        self.assertTrue(any("History" in line for line in logs), logs)

    def test_n9b_recovery_leaves_the_spool_untouched_while_history_is_unreadable(self):
        self.die_with_a_spooled_capture()
        before = {path.name: path.read_bytes() for path in spooled(self.base)}
        (self.base / "history.jsonl").write_text("{damaged\n", encoding="utf-8")
        store, coordinator = open_history(self.base, tolerate_unreadable=True)
        self.assertIsNotNone(store.unreadable)
        logs: list[str] = []
        self.assertEqual(recover(new_spool(self.base), store, coordinator, logs), 1)
        self.assertEqual({path.name: path.read_bytes() for path in spooled(self.base)}, before)
        self.assertEqual(len([line for line in logs if "wait" in line]), 1, logs)

    def test_n9b_a_capture_whose_row_was_saved_just_before_the_stop_is_not_added_twice(self):
        capture_id, captured_ts, raw = self.die_with_a_spooled_capture()
        store, coordinator = open_history(self.base)
        from audio_codec import prepare_canonical
        coordinator.append_live("Please call me back at four",
                                prepare_canonical(scrivo.prepare_for_whisper(raw, 48000.0)), 1.2,
                                WHISPER.model_repo, ts=captured_ts, raw_samples=raw,
                                raw_sample_rate=48000.0, provenance="live", adaptive=False)
        recover(new_spool(self.base), store, coordinator, [])
        self.assertEqual([row["text"] for row in store.entries(10)], ["Please call me back at four"])
        self.assertEqual(spooled(self.base), [])

    def test_n9b_a_partial_write_cut_off_by_the_stop_is_removed(self):
        folder = self.base / "capture-spool"
        folder.mkdir(mode=0o700)
        partial = folder / f".{uuid.uuid4().hex}-1.wav.{uuid.uuid4().hex}.tmp"
        partial.write_bytes(b"RIFF")
        store, coordinator = open_history(self.base)
        recover(new_spool(self.base), store, coordinator, [])
        self.assertEqual(spooled(self.base), [])
        self.assertEqual(store.entries(10), [])


def closures(names, namespace):
    """Compile the named functions defined in scrivo.run() into ``namespace``,
    whose entries stand in for the closure variables they read (as in
    test_mac_lifecycle)."""
    source = Path(scrivo.__file__).read_text(encoding="utf-8")
    run = next(node for node in ast.parse(source).body
               if isinstance(node, ast.FunctionDef) and node.name == "run")
    found = {node.name: node for node in ast.walk(run) if isinstance(node, ast.FunctionDef)}
    module = ast.Module(body=[found[name] for name in names], type_ignores=[])
    exec(compile(module, scrivo.__file__, "exec"), namespace)
    return namespace


@unittest.skipUnless(sys.platform == "darwin", "the Mac key-up and worker — macOS-only")
class MacKeyUpTests(TemporaryData):
    def finish(self, raw):
        jobs: queue.Queue = queue.Queue()
        spool = new_spool(self.base)
        capture = types.SimpleNamespace(end=lambda include_stream=False: (raw, None),
                                        release_soon=lambda: None, native_rate=48000.0)
        namespace = {"shutdown": scrivo.ShutdownBoundary(), "capture": capture, "log": lambda line: None,
                     "status_ui": None, "jobs": jobs, "time": time, "uuid": uuid, "spool": spool,
                     "current_speech_config": lambda: WHISPER,
                     "worker": types.SimpleNamespace(keep_untranscribed=lambda job: None)}
        closures(["finish_capture"], namespace)["finish_capture"]()
        return jobs, spool

    def test_n9b_a_finished_capture_is_spooled_before_it_is_transcribed(self):
        raw = speech(1.0)
        jobs, spool = self.finish(raw)
        job = jobs.get_nowait()  # queued, and no worker ever runs it: the process dies here
        self.assertTrue(wait_for(lambda: spooled(self.base)), "the key-up spooled nothing")
        self.assertTrue(spooled(self.base)[0].name.startswith(f"{job[5]}-"), spooled(self.base))

    def test_n9b_a_tap_is_never_spooled(self):
        jobs, spool = self.finish(speech(0.1))
        self.assertTrue(jobs.empty())
        time.sleep(0.05)
        self.assertEqual(spooled(self.base), [])


@unittest.skipUnless(sys.platform == "darwin", "the Mac transcription worker — macOS-only")
class MacWorkerReleaseTests(TemporaryData):
    def harness(self, behaviour=None, append=None):
        import test_transcription_worker as golden
        self.golden = golden
        h = golden.Harness(behaviour or golden.says("Please call me back at four"), append=append)
        h.worker.spool = self.spool = new_spool(self.base)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        import dictionary
        import settings
        from unittest.mock import patch
        for target, name in ((dictionary, "DICTIONARY_PATH"), (settings, "SETTINGS_PATH")):
            patcher = patch.object(target, name, Path(folder.name) / f"{name.lower()}.txt")
            patcher.start()
            self.addCleanup(patcher.stop)
        return h

    def spooled_job(self, h, raw):
        job = h.live_job(raw)
        job = job[:5] + (uuid.uuid4().hex,) + job[6:]
        self.spool.submit(job[5], raw, job[2], job[3])
        return job

    def settled(self):
        self.assertTrue(wait_for(lambda: writer_idle(self.spool)))
        return spooled(self.base)

    def test_n9b_a_successful_dictation_leaves_no_spool(self):
        h = self.harness()
        h.run_jobs(self.spooled_job(h, self.golden.speech(2.0)))
        self.assertEqual(h.injected, ["Please call me back at four"])
        self.assertEqual(self.settled(), [])

    def test_n9b_a_dropped_tap_leaves_no_spool(self):
        h = self.harness()
        h.run_jobs(self.spooled_job(h, self.golden.speech(0.1)))
        self.assertEqual(h.coordinator.appended, [])
        self.assertEqual(self.settled(), [])

    def test_n9b_a_capture_kept_at_shutdown_leaves_one_row_and_no_spool(self):
        h = self.harness()
        job = self.spooled_job(h, self.golden.speech(2.0))
        h.shutdown.request()
        h.worker.keep_untranscribed(job)
        self.assertEqual([row["text"] for row in h.coordinator.appended], [NOT_TRANSCRIBED])
        self.assertEqual(self.settled(), [])

    def test_n9b_a_keep_history_could_not_save_stays_spooled_for_the_next_launch(self):
        def disk_full(*args, **kwargs):
            raise OSError(28, "No space left on device")
        h = self.harness(append=disk_full)
        job = self.spooled_job(h, self.golden.speech(2.0))
        h.shutdown.request()
        h.worker.keep_untranscribed(job)
        self.assertEqual(len(self.settled()), 1, "audio no row holds was deleted")

    def test_n9b_a_keep_read_only_history_did_not_save_stays_spooled(self):
        def read_only(text, prepared, seconds, model, **kw):
            return {"id": "row1", "text": text, "saved": False}
        h = self.harness(append=read_only)
        job = self.spooled_job(h, self.golden.speech(2.0))
        h.shutdown.request()
        h.worker.keep_untranscribed(job)
        self.assertEqual(len(self.settled()), 1, "audio no row holds was deleted")

    def test_n9b_delivered_text_whose_row_failed_is_not_recovered_later(self):
        # F16a: the text reached the cursor; a recovered row would be a duplicate.
        def disk_full(*args, **kwargs):
            raise OSError(28, "No space left on device")
        h = self.harness(append=disk_full)
        h.run_jobs(self.spooled_job(h, self.golden.speech(2.0)))
        self.assertEqual(h.injected, ["Please call me back at four"])
        self.assertEqual(self.settled(), [])



@unittest.skipUnless(sys.platform == "win32", "the Windows app — Windows-only")
class WindowsAppTests(TemporaryData):
    """scrivo_win.run end to end with the fake key, microphone and model of test_scrivo_win."""

    def run_app(self, drive, whisper):
        import contextlib
        import scrivo_win
        import test_scrivo_win as win_tests
        hooks, boundaries, controllers, logs, delivered = [], [], [], [], []
        rate = 16_000
        voiced = (np.sin(np.arange(rate) / 3) * 0.2).astype(np.float32)
        _, FakeCapture, FakeHook = win_tests._app_fakes(rate, [voiced, voiced.copy()], [], hooks, [])
        results: dict = {}

        def driver():
            try:
                win_tests._wait_for(lambda: controllers and hooks, "listening")
                drive(hooks[0], results)
            except BaseException as exc:
                results["error"] = exc
            finally:
                boundaries[0].request()  # Scrivo ends without its drain

        with contextlib.ExitStack() as stack:
            win_tests._patched_run(stack, self.base, (whisper, FakeCapture, FakeHook), logs=logs,
                                   boundaries=boundaries, controllers=controllers)
            stack.enter_context(unittest.mock.patch.object(
                scrivo_win.win_inject, "deliver", lambda text, **kwargs: delivered.append(text)))
            stack.enter_context(unittest.mock.patch.object(scrivo_win.win_inject, "copy_text", delivered.append))
            thread = threading.Thread(target=driver, daemon=True)
            thread.start()
            scrivo_win.run("right-ctrl", "auto", None, None, None, 0.0, "cpu")
            thread.join(10)
        self.assertNotIn("error", results, (results, logs))
        return results, logs

    def test_n9b_a_capture_the_app_stopped_on_is_in_history_at_the_next_launch(self):
        from history import HistoryStore
        calls, blocked, unblock = [], threading.Event(), threading.Event()
        self.addCleanup(unblock.set)

        class Whisper:
            def __init__(self, model_dir, *, device, log):
                self.device, self.compute_type = "cpu", "int8"

            def transcribe(self, samples, **kwargs):
                calls.append(len(samples))
                if len(calls) == 3:  # warmup, the first dictation, then this one
                    blocked.set()
                    unblock.wait(WAIT_S * 3)
                result = {"text": "" if len(calls) == 1 else "Please call me back"}
                if kwargs.get("language") is None and kwargs.get("allowed_languages"):
                    result["language"] = tuple(kwargs["allowed_languages"])[0]
                return result

        def first_session(hook, results):
            import test_scrivo_win as win_tests
            for _ in range(2):
                hook.engine.pressed()
                time.sleep(0.45)
                hook.engine.released()
                if not HistoryStore(self.base).entries(1):
                    win_tests._wait_for(lambda: HistoryStore(self.base).entries(1), "the first row")
            win_tests._wait_for(blocked.is_set, "the model on the second capture")
            win_tests._wait_for(lambda: [p for p in spooled(self.base) if p.suffix == ".wav"], "its spool")
            results["spooled"] = spooled(self.base)

        results, logs = self.run_app(first_session, Whisper)
        self.assertEqual(len(results["spooled"]), 1, "the saved dictation stayed spooled, or the other one did not")
        self.assertEqual([row["text"] for row in HistoryStore(self.base).entries(10)], ["Please call me back"])

        results, logs = self.run_app(lambda hook, results: None, Whisper)
        self.assertEqual([row["text"] for row in HistoryStore(self.base).entries(10)],
                         [NOT_TRANSCRIBED, "Please call me back"], logs)
        self.assertEqual(spooled(self.base), [])


if __name__ == "__main__":
    unittest.main()
