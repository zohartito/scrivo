"""N9b: a finished capture is on disk from key-up until its History row exists.

Before N9b a recording lived only in RAM between key-up and its History row,
so a crash, a SIGKILL or a logout that skipped the drain lost it. Now the
key-up hands the audio to ``history.CaptureSpool`` (written off the event-tap
thread), the worker or teardown deletes the file once the capture's row
exists or its fate is decided, and the next launch turns any file still there
into a "[not transcribed: Sotto stopped]" row, exactly like the N9 keep.
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

import sotto
from audio_codec import encode_pcm16
from speech_config import resolve_speech_config

ROOT = Path(__file__).resolve().parents[1]
WHISPER = resolve_speech_config("auto", language="en")
NOT_TRANSCRIBED = "[not transcribed: Sotto stopped]"
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
    return store, sotto.history_coordinator(store, learning)


def recover(spool, store, coordinator, logs):
    return sotto.recover_capture_spool(spool, store, coordinator, model=WHISPER.model_repo,
                                       speech_config=WHISPER, glossary_terms=(), log=logs.append)


class TemporaryData(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix="sotto-spool-")
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
        self.assertIsNone(spool._writer, "an idle Sotto started the spool writer")

    def test_n9b_a_second_sotto_on_the_same_data_folder_neither_spools_nor_recovers(self):
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
        env = dict(os.environ, SOTTO_DATA_DIR=str(self.base), SOTTO_HF_HOME=str(self.base / "hf"))
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
                                prepare_canonical(sotto.prepare_for_whisper(raw, 48000.0)), 1.2,
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
    """Compile the named functions defined in sotto.run() into ``namespace``,
    whose entries stand in for the closure variables they read (as in
    test_mac_lifecycle)."""
    source = Path(sotto.__file__).read_text(encoding="utf-8")
    run = next(node for node in ast.parse(source).body
               if isinstance(node, ast.FunctionDef) and node.name == "run")
    found = {node.name: node for node in ast.walk(run) if isinstance(node, ast.FunctionDef)}
    module = ast.Module(body=[found[name] for name in names], type_ignores=[])
    exec(compile(module, sotto.__file__, "exec"), namespace)
    return namespace


@unittest.skipUnless(sys.platform == "darwin", "the Mac key-up and worker — macOS-only")
class MacKeyUpTests(TemporaryData):
    def finish(self, raw):
        jobs: queue.Queue = queue.Queue()
        spool = new_spool(self.base)
        capture = types.SimpleNamespace(end=lambda include_stream=False: (raw, None),
                                        release_soon=lambda: None, native_rate=48000.0)
        namespace = {"shutdown": sotto.ShutdownBoundary(), "capture": capture, "log": lambda line: None,
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
    """sotto_win.run end to end with the fake key, microphone and model of test_sotto_win."""

    def run_app(self, drive, whisper):
        import contextlib
        import sotto_win
        import test_sotto_win as win_tests
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
                boundaries[0].request()  # Sotto ends without its drain

        with contextlib.ExitStack() as stack:
            win_tests._patched_run(stack, self.base, (whisper, FakeCapture, FakeHook), logs=logs,
                                   boundaries=boundaries, controllers=controllers)
            stack.enter_context(unittest.mock.patch.object(
                sotto_win.win_inject, "deliver", lambda text, **kwargs: delivered.append(text)))
            stack.enter_context(unittest.mock.patch.object(sotto_win.win_inject, "copy_text", delivered.append))
            thread = threading.Thread(target=driver, daemon=True)
            thread.start()
            sotto_win.run("right-ctrl", "auto", None, None, None, 0.0, "cpu")
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
            import test_sotto_win as win_tests
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
