"""The menu-bar actions of the Mac app (History, settings, lifecycle).

Each test drives one action with fakes (no AppKit main loop, no clipboard,
no model, no mic, no threads: background work runs inline and is recorded
as a "thread" event) and checks everything it does: History and learning
calls, deliveries, queued jobs, UI calls and log lines.

The expected values were recorded from the actions as closures inside
scrivo.run(), before they moved into AppController; the class reproduces
them exactly.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

if sys.platform != "darwin":
    raise unittest.SkipTest("the Mac menu actions — macOS-only")

import dictionary
import login_item
import nemotron_backend
import offline_runtime
import settings
import settings_window
import scrivo
import scrivo_paths
import updates
from app_controller import AppController

CONFIG = "speech-config"


class World:
    """The collaborators an action reaches, as fakes that record each call
    in ``events`` (in order)."""

    def __init__(self, *, rows=(), adaptive=None, closed=None, stopping=False):
        self.events: list = []
        self.rows = {row["id"]: dict(row) for row in rows}
        self.shutdown = scrivo.ShutdownBoundary()
        self.gate = scrivo.CaptureGate()
        self.busy = False
        self.lifecycle = scrivo.Lifecycle(self.shutdown, self.gate, end_recording=lambda: None,
                                         busy=lambda: self.busy)
        if closed:
            self.assert_true(self.lifecycle.close(closed))
        if stopping:
            self.shutdown.request()
        self.jobs: queue.Queue = queue.Queue()
        self.pending = scrivo.PendingDeliveries()
        self.status_ui = Recorder(self.events)
        self.adaptive_runtime = adaptive
        self.coordinator = Coordinator(self.events)
        self.dependency_guard = object()
        self.store = Store(self)
        self.fail: dict = {}  # helper name -> exception it raises
        # Settings: how this run was started, and its live bindings.
        self.adaptive = False
        self.use_nemotron = False
        self.use_parakeet = False
        self.engine_switching = True
        self.active_engine = "whisper"
        self.settings_locked = frozenset()
        self.binding = {"trigger": "right-option", "hotkey_spec": "ctrl-opt-d"}
        self.language_lock = threading.Lock()
        self.language_state = {"mode": "auto"}
        self.restart = types.SimpleNamespace(request=self.restart_request)
        self.restart_failure = None   # the message a refused restart reports
        self.restart_error = None     # what the restart request raises
        self.restart_accepts = True
        # The gesture engine and capture the lifecycle actions reach.
        self.recording = False
        self.capture_active = False
        self.start_ok = True
        self.engine = types.SimpleNamespace(force_finish=self.force_finish, force_start=self.force_start)
        self.capture = types.SimpleNamespace(is_active=lambda: self.capture_active)
        # The update drain: what wait_until_idle answers.
        self.drained = True
        # The commit this run started from (run() reads it once at launch).
        self.started_from = "c2e7f0f" + "0" * 33
        self.parakeet_ready = True
        self.persist_failure = None   # the restart's on_failure message
        self.installer_result = (True, "done")

    @staticmethod
    def assert_true(value):
        assert value

    # -- helpers run() hands over --------------------------------------------
    def ui_call(self, method, *args):
        """Records the call; runs Scrivo's own main-thread functions (the
        menu bar's methods are only recorded)."""
        self.events.append(("ui", method.__name__, *args))
        if not getattr(method, "recorded", False) and method.__name__ != "refresh":
            method(*args)

    def deliver_call(self, method, *args):
        self.pending.add()
        self.events.append(("deliver", *args))
        method(*args)

    def log(self, line):
        self.events.append(("log", line))

    def start_thread(self, target):
        self.events.append(("thread",))
        target()

    def set_clipboard(self, text):
        self.events.append(("clipboard", text))

    def current_speech_config(self):
        return CONFIG

    def active_learning_history_ids(self, coordinator):
        assert coordinator is self.coordinator
        return {"active-row"}

    def _helper(self, name, *args, result=None, **kwargs):
        self.events.append((name, *args, *([kwargs] if kwargs else [])))
        if name in self.fail:
            raise self.fail[name]
        return result

    def nonadaptive_correct(self, guard, coordinator, entry_id, text, *, expected_revision=None):
        assert (guard, coordinator) == (self.dependency_guard, self.coordinator)
        found = entry_id in self.rows
        return self._helper("nonadaptive_correct", entry_id, text, expected_revision=expected_revision,
                            result={"id": entry_id} if found else None)

    def nonadaptive_delete(self, guard, coordinator, entry_id):
        assert (guard, coordinator) == (self.dependency_guard, self.coordinator)
        return self._helper("nonadaptive_delete", entry_id, result=True)

    def nonadaptive_clear(self, guard, coordinator):
        assert (guard, coordinator) == (self.dependency_guard, self.coordinator)
        return self._helper("nonadaptive_clear")

    def nonadaptive_revoke_learning(self, guard, coordinator, entry_id):
        assert (guard, coordinator) == (self.dependency_guard, self.coordinator)
        return self._helper("nonadaptive_revoke_learning", entry_id, result=2)

    def save_audio_copy(self, source, folder, timestamp):
        self._helper("save_audio_copy", source, folder, timestamp)
        return folder / "scrivo-20261010-120000.wav"

    def save_language_mode(self, mode):
        return self._helper("save_language_mode", mode, result=f"canonical-{mode}")

    def parakeet_cached(self):
        return self.parakeet_ready

    def persist_engine_and_restart(self, mode, request, on_failure=None):
        assert request is self.restart.request
        self._helper("persist_engine_and_restart", mode, gate_open(self.gate))
        if self.persist_failure:
            on_failure(self.persist_failure)

    def rebind(self, new_trigger=None, new_hotkey_spec=None):
        self._helper("rebind", new_trigger, new_hotkey_spec)

    def run_installer(self, argv):
        self._helper("run_installer", argv)
        return self.installer_result

    def restart_request(self, on_failure=None):
        self.events.append(("restart.request",))
        if self.restart_error is not None:
            raise self.restart_error
        if self.restart_failure:
            on_failure(self.restart_failure)
        return self.restart_accepts

    def force_finish(self, finish=None):
        self.events.append(("engine.force_finish", finish == self.on_finish))
        return self.recording

    def force_start(self):
        self.events.append(("engine.force_start",))
        return self.start_ok

    def on_finish(self):
        self.events.append(("on_finish",))

    def wait_until_idle(self, busy, *, deadline_s):
        self.events.append(("wait_until_idle", busy(), deadline_s))
        return self.drained


def gate_open(gate) -> bool:
    with gate.starting() as allowed:
        return allowed


class Recorder:
    """status_ui: every method is a named no-op the World records through
    ui_call (or, called directly on the main thread, as "direct"); ``ask``
    answers with ``answer``."""

    def __init__(self, events):
        self._events = events
        self.answer = False

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def method(*args):
            self._events.append(("direct", name, *args))
            return self.answer if name == "ask" else None
        method.__name__ = name
        method.recorded = True
        return method


class Store:
    def __init__(self, world):
        self.world = world

    def get(self, entry_id):
        if "store.get" in self.world.fail:
            raise self.world.fail["store.get"]
        return self.world.rows.get(entry_id)

    def entries(self, limit):
        self.world.events.append(("entries", limit))
        return list(self.world.rows.values())

    def audio_path(self, entry_id):
        return Path("/history/audio") / f"{entry_id}.wav"


class Coordinator:
    def __init__(self, events):
        self.events = events
        self.snapshot = object()

    def enroll(self, entry_id):
        self.events.append(("enroll", entry_id))
        return self.snapshot


class Adaptive:
    """An adaptive runtime: review state, a stage and the review calls."""

    def __init__(self, events, *, eligible=True, status=None, review_states=()):
        self.events = events
        self.eligible = eligible
        self._status = status or {"active_generation": None,
                                  "development": {"ready": False, "verified_nonempty": 3}}
        self.review_states = list(review_states)

    def review_list(self):
        return self.review_states

    def review_eligible_entry(self, row):
        return self.eligible and row is not None

    def status(self):
        return self._status

    def review(self, entry_id, decision, **kwargs):
        self.events.append(("review", entry_id, decision, kwargs))

    def delete(self, entry_id):
        self.events.append(("adaptive.delete", entry_id))

    def clear(self):
        self.events.append(("adaptive.clear",))

    def revoke(self, entry_id):
        self.events.append(("adaptive.revoke", entry_id))
        return 4


ROW = {"id": "row1", "text": "hello there", "ts": 1_760_000_000.0}


# -- the controller, built over the World --------------------------------------

def build(world: World) -> AppController:
    """An AppController whose collaborators are the World's fakes."""
    return AppController(
        shutdown=world.shutdown, lifecycle=world.lifecycle, jobs=world.jobs, store=world.store,
        coordinator=world.coordinator, dependency_guard=world.dependency_guard,
        adaptive_runtime=world.adaptive_runtime, status_ui=world.status_ui, ui_call=world.ui_call,
        deliver_call=world.deliver_call, pending_deliveries=world.pending, set_clipboard=world.set_clipboard,
        current_speech_config=world.current_speech_config, log=world.log, history_keep=scrivo.HISTORY_KEEP,
        overlay_learning_state=scrivo.overlay_learning_state,
        active_learning_history_ids=world.active_learning_history_ids,
        nonadaptive_correct=world.nonadaptive_correct, nonadaptive_delete=world.nonadaptive_delete,
        nonadaptive_clear=world.nonadaptive_clear, nonadaptive_revoke_learning=world.nonadaptive_revoke_learning,
        save_audio_copy=world.save_audio_copy, adaptive=world.adaptive, use_nemotron=world.use_nemotron,
        use_parakeet=world.use_parakeet, engine_switching=world.engine_switching,
        active_engine=world.active_engine, settings_locked=world.settings_locked, binding=world.binding,
        rebind=world.rebind, restart=world.restart, language_lock=world.language_lock,
        language_state=world.language_state, save_language_mode=world.save_language_mode,
        parakeet_cached=world.parakeet_cached, persist_engine_and_restart=world.persist_engine_and_restart,
        run_installer=world.run_installer, triggers=scrivo.TRIGGERS, trigger_labels=scrivo.TRIGGER_LABELS,
        scrivo_script=Path(scrivo.__file__).resolve(), engine=world.engine, capture=world.capture,
        on_finish=world.on_finish, started_from=world.started_from, start_thread=world.start_thread,
        wait_until_idle=world.wait_until_idle)


# -- History ------------------------------------------------------------------

class RefreshHistoryTests(unittest.TestCase):
    def test_rows_get_the_learning_overlay(self):
        world = World(rows=[ROW, {"id": "active-row", "text": "x", "ts": 2.0}])
        build(world).refresh_history()
        self.assertEqual(world.events, [
            ("entries", scrivo.HISTORY_KEEP),
            ("ui", "refresh_history", [dict(ROW), {"id": "active-row", "text": "x", "ts": 2.0,
                                                   "learning_state": "active"}]),
        ])

    def test_without_a_menu_bar_nothing_happens(self):
        world = World(rows=[ROW])
        world.status_ui = None
        build(world).refresh_history()
        self.assertEqual(world.events, [])

    def test_adaptive_rows_get_review_state_and_the_stage_label(self):
        cases = [
            ({"active_generation": {"status": "evaluating"}, "development": {}}, "adaptive: evaluating"),
            ({"active_generation": {"status": "pool_closed"}, "development": {}}, "adaptive: evaluating"),
            ({"active_generation": {"status": "open", "resolved": 7}, "development": {}},
             "adaptive: pool 7/33"),
            ({"active_generation": None, "development": {"ready": True}}, "adaptive: collecting challenger"),
            ({"active_generation": None, "development": {"verified_nonempty": 3}}, "adaptive: dev 3/20"),
        ]
        for status, label in cases:
            with self.subTest(label=label):
                world = World(rows=[ROW])
                world.adaptive_runtime = Adaptive(world.events, status=status,
                                                  review_states=[{"history_id": "row1", "status": "pending"}])
                build(world).refresh_history()
                self.assertEqual(world.events, [
                    ("entries", scrivo.HISTORY_KEEP),
                    ("ui", "set_adaptive_stage", label),
                    ("ui", "refresh_history", [dict(ROW, adaptive_eligible=True, review_state="pending")]),
                ])


class CopyTests(unittest.TestCase):
    def test_copy_puts_the_transcript_on_the_clipboard_through_the_counted_delivery(self):
        world = World(rows=[ROW])
        build(world).copy("row1")
        self.assertEqual(world.events, [("thread",), ("deliver", "hello there"), ("clipboard", "hello there")])
        self.assertEqual(world.pending.count(), 0, "the copy did not finish its PendingDeliveries entry")

    def test_a_missing_row_copies_nothing(self):
        world = World()
        build(world).copy("gone")
        self.assertEqual(world.events, [("thread",)])

    def test_a_store_error_is_shown(self):
        world = World(rows=[ROW])
        world.fail["store.get"] = OSError("disk")
        build(world).copy("row1")
        self.assertEqual(world.events, [("thread",), ("ui", "show_error", "Could not copy transcript", "disk")])

    def test_after_shutdown_the_clipboard_is_left_alone_but_the_count_finishes(self):
        world = World()
        actions = build(world)
        world.pending.add()
        world.shutdown.request()
        actions.copy_text("late")
        self.assertEqual(world.events, [])
        self.assertEqual(world.pending.count(), 0)


class RetryTests(unittest.TestCase):
    def test_retry_queues_a_job_with_the_current_speech_settings(self):
        world = World()
        build(world).retry("row1")
        job = world.jobs.get_nowait()
        self.assertEqual(job[:2], ("retry", "row1"))
        self.assertIsInstance(job[2], float)
        self.assertRegex(job[3], r"^[0-9a-f]{32}$")
        self.assertEqual(job[4], CONFIG)
        self.assertEqual(len(job), 5)
        self.assertEqual(world.events, [])

    def test_retry_is_refused_while_the_gate_is_closed(self):
        world = World(closed="updating")
        build(world).retry("row1")
        self.assertTrue(world.jobs.empty())
        self.assertEqual(world.events, [
            ("log", "○ Scrivo is updating — Retry refused"),
            ("ui", "show_error", "Not retried", "Scrivo is updating. Retry once it is running again."),
        ])

    def test_retry_is_refused_while_stopping(self):
        world = World(stopping=True)
        build(world).retry("row1")
        self.assertTrue(world.jobs.empty())
        self.assertEqual(world.events[0], ("log", "○ Scrivo is stopping — Retry refused"))


class SaveAudioTests(unittest.TestCase):
    def test_save_copies_the_audio_to_the_desktop(self):
        world = World(rows=[ROW])
        build(world).save("row1")
        self.assertEqual(world.events, [
            ("thread",),
            ("save_audio_copy", Path("/history/audio/row1.wav"), Path.home() / "Desktop", ROW["ts"]),
            ("log", "✓ saved audio to Desktop as scrivo-20261010-120000.wav"),
        ])

    def test_a_missing_row_saves_nothing(self):
        world = World()
        build(world).save("gone")
        self.assertEqual(world.events, [("thread",)])

    def test_missing_audio_says_so(self):
        world = World(rows=[ROW])
        world.fail["save_audio_copy"] = FileNotFoundError("gone")
        build(world).save("row1")
        self.assertEqual(world.events[2:], [
            ("log", "! audio missing for that entry"),
            ("ui", "show_error", "Could not save audio", "The audio file is no longer available."),
        ])

    def test_another_error_is_shown(self):
        world = World(rows=[ROW])
        world.fail["save_audio_copy"] = PermissionError("Desktop is read-only")
        build(world).save("row1")
        self.assertEqual(world.events[2:], [("ui", "show_error", "Could not save audio", "Desktop is read-only")])


class DeleteAndClearTests(unittest.TestCase):
    def test_delete_revokes_then_refreshes(self):
        world = World(rows=[ROW])
        build(world).delete("row1")
        self.assertEqual(world.events, [
            ("thread",), ("nonadaptive_delete", "row1"),
            ("entries", scrivo.HISTORY_KEEP), ("ui", "refresh_history", [dict(ROW)]),
        ])

    def test_adaptive_delete_goes_through_the_runtime(self):
        world = World()
        world.adaptive_runtime = Adaptive(world.events)
        build(world).delete("row1")
        self.assertEqual(world.events[:2], [("thread",), ("adaptive.delete", "row1")])

    def test_a_delete_error_is_shown(self):
        world = World()
        world.fail["nonadaptive_delete"] = ValueError("locked")
        build(world).delete("row1")
        self.assertEqual(world.events[-1], ("ui", "show_error", "Could not delete recording", "locked"))

    def test_clear(self):
        world = World()
        build(world).clear()
        self.assertEqual(world.events, [("thread",), ("nonadaptive_clear",),
                                        ("entries", scrivo.HISTORY_KEEP), ("ui", "refresh_history", [])])
        world = World()
        world.adaptive_runtime = Adaptive(world.events)
        build(world).clear()
        self.assertEqual(world.events[:2], [("thread",), ("adaptive.clear",)])
        world = World()
        world.fail["nonadaptive_clear"] = OSError("disk full")
        build(world).clear()
        self.assertEqual(world.events[-1], ("ui", "show_error", "Could not clear history", "disk full"))

    def test_nothing_changes_once_shutdown_is_requested(self):
        for name, args in (("delete", ("row1",)), ("clear", ()), ("correct", ("row1", "x")),
                           ("enroll", ("row1",)), ("revoke", ("row1",)), ("correct_as_is", ("row1",)),
                           ("no_speech", ("row1",)), ("skip_learning_review", ("row1", "noise"))):
            with self.subTest(action=name):
                world = World(rows=[ROW], stopping=True)
                world.adaptive_runtime = Adaptive(world.events)
                getattr(build(world), name)(*args)
                self.assertEqual(world.events, [("thread",)])


class CorrectTests(unittest.TestCase):
    def correct(self, world, *, suggestions=()):
        with patch.object(dictionary, "load", return_value=["existing"]) as load, \
             patch.object(dictionary, "suggest", return_value=list(suggestions)) as suggest:
            build(world).correct("row1", "hello their", 3)
        return load, suggest

    def test_a_correction_is_saved_and_reported(self):
        world = World(rows=[ROW])
        _, suggest = self.correct(world)
        suggest.assert_called_once_with("hello there", "hello their", ["existing"])
        self.assertEqual(world.events, [
            ("thread",), ("nonadaptive_correct", "row1", "hello their", {"expected_revision": 3}),
            ("entries", scrivo.HISTORY_KEEP), ("ui", "refresh_history", [dict(ROW)]),
            ("ui", "show_info", "Correction saved",
             "The corrected transcript was saved. Add it to the learning set separately if wanted."),
        ])

    def test_dictionary_suggestions_are_offered(self):
        world = World(rows=[ROW])
        self.correct(world, suggestions=[dictionary.Rule("there", "their")])
        self.assertEqual(world.events[-1], (
            "ui", "offer_dictionary_rules", [("there", "their")],
            "The corrected transcript was saved. Add it to the learning set separately if wanted."))

    def test_an_eligible_adaptive_row_is_reviewed(self):
        world = World(rows=[ROW])
        world.adaptive_runtime = Adaptive(world.events)
        self.correct(world)
        self.assertEqual(world.events[1], ("review", "row1", "corrected",
                                           {"reference": "hello their", "expected_revision": 3}))
        self.assertEqual(world.events[-1][-1],
                         "The corrected transcript and audio are retained locally for private learning.")

    def test_an_ineligible_adaptive_row_is_corrected_normally(self):
        world = World(rows=[ROW])
        world.adaptive_runtime = Adaptive(world.events, eligible=False)
        self.correct(world)
        self.assertEqual(world.events[1][0], "nonadaptive_correct")

    def test_a_deleted_row_is_reported(self):
        world = World()
        self.correct(world)
        self.assertEqual(world.events, [
            ("thread",), ("nonadaptive_correct", "row1", "hello their", {"expected_revision": 3}),
            ("ui", "show_error", "Could not save correction", "The recording no longer exists"),
        ])


class AddRulesTests(unittest.TestCase):
    def test_confirmed_rules_are_added_and_history_refreshed(self):
        world = World()
        with patch.object(dictionary, "add", return_value=1) as add:
            build(world).add_rules([("there", "their")])
        self.assertEqual(add.call_args.args, ([dictionary.Rule("there", "their")], dictionary.DICTIONARY_PATH))
        self.assertEqual(world.events, [("log", "dictionary: added 1 rule(s)"),
                                        ("entries", scrivo.HISTORY_KEEP), ("ui", "refresh_history", [])])

    def test_an_error_is_shown(self):
        world = World()
        with patch.object(dictionary, "add", side_effect=OSError("read-only")):
            build(world).add_rules([("there", "their")])
        self.assertEqual(world.events, [("ui", "show_error", "Could not update the dictionary", "read-only")])


class LearningTests(unittest.TestCase):
    def test_enroll(self):
        world = World()
        build(world).enroll("row1")
        self.assertEqual(world.events, [
            ("thread",), ("enroll", "row1"), ("entries", scrivo.HISTORY_KEEP), ("ui", "refresh_history", []),
            ("ui", "show_info", "Added to learning set",
             "The corrected audio and transcript are now retained locally."),
        ])

    def test_enroll_without_a_correction_is_refused(self):
        world = World()
        world.coordinator.snapshot = None
        build(world).enroll("row1")
        self.assertEqual(world.events[-1], ("ui", "show_error", "Could not enroll sample",
                                            "A current correction is required before enrollment"))

    def test_revoke(self):
        world = World()
        build(world).revoke("row1")
        self.assertEqual(world.events, [
            ("thread",), ("nonadaptive_revoke_learning", "row1"), ("entries", scrivo.HISTORY_KEEP),
            ("ui", "refresh_history", []),
            ("ui", "show_info", "Removed from learning set", "Removed 2 local learning sample(s)."),
        ])
        world = World()
        world.adaptive_runtime = Adaptive(world.events)
        build(world).revoke("row1")
        self.assertEqual(world.events[1], ("adaptive.revoke", "row1"))
        self.assertEqual(world.events[-1][-1], "Removed 4 local learning sample(s).")

    def test_reviews_need_the_adaptive_runtime(self):
        for name, args in (("correct_as_is", ("row1", 5)), ("no_speech", ("row1", 5)),
                           ("skip_learning_review", ("row1", "noise", 5))):
            with self.subTest(action=name):
                world = World()
                getattr(build(world), name)(*args)
                self.assertEqual(world.events, [("thread",), ("ui", "show_error", "Could not review recording",
                                                              "Adaptive English is not enabled")])

    def test_reviews_reach_the_adaptive_runtime_then_refresh(self):
        cases = (("correct_as_is", ("row1", 5), ("review", "row1", "correct_as_is", {"expected_revision": 5})),
                 ("no_speech", ("row1", 5), ("review", "row1", "no_speech", {"expected_revision": 5})),
                 ("skip_learning_review", ("row1", "noise", 5),
                  ("review", "row1", "skip", {"reason": "noise", "expected_revision": 5})))
        for name, args, review in cases:
            with self.subTest(action=name):
                world = World()
                world.adaptive_runtime = Adaptive(world.events, status={"active_generation": None,
                                                                        "development": {"ready": True}})
                getattr(build(world), name)(*args)
                self.assertEqual(world.events, [("thread",), review, ("entries", scrivo.HISTORY_KEEP),
                                                ("ui", "set_adaptive_stage", "adaptive: collecting challenger"),
                                                ("ui", "refresh_history", [])])


# -- Settings -------------------------------------------------------------------

class LanguageTests(unittest.TestCase):
    def test_the_choice_is_saved_shown_and_used_by_the_next_recording(self):
        world = World()
        build(world).set_language("en")
        self.assertEqual(world.language_state, {"mode": "canonical-en"})
        self.assertEqual(world.events, [
            ("save_language_mode", "en"), ("ui", "set_language_mode", "canonical-en", True),
            ("log", "language: canonical-en (applies to the next recording)"),
        ])

    def test_engines_without_a_language_choice_ignore_it(self):
        for flag in ("adaptive", "use_nemotron", "use_parakeet"):
            with self.subTest(flag=flag):
                world = World()
                setattr(world, flag, True)
                build(world).set_language("en")
                self.assertEqual((world.events, world.language_state), ([], {"mode": "auto"}))

    def test_a_save_error_is_shown_and_changes_nothing(self):
        world = World()
        world.fail["save_language_mode"] = OSError("read-only")
        build(world).set_language("he")
        self.assertEqual(world.language_state, {"mode": "auto"})
        self.assertEqual(world.events[-1], ("ui", "show_error", "Could not change language", "read-only"))


class EngineTests(unittest.TestCase):
    def test_switching_closes_the_gate_then_persists_and_restarts(self):
        world = World()
        build(world).set_engine("parakeet")
        self.assertEqual(world.events, [("persist_engine_and_restart", "parakeet", False)])
        self.assertEqual(world.lifecycle.closed, "switching engines")

    def test_nothing_happens_for_the_current_engine_adaptive_or_a_locked_engine(self):
        for flag, value in (("active_engine", "parakeet"), ("adaptive", True), ("engine_switching", False)):
            with self.subTest(flag=flag):
                world = World()
                setattr(world, flag, value)
                build(world).set_engine("parakeet")
                self.assertEqual(world.events, [])
                self.assertIsNone(world.lifecycle.closed)

    def test_a_busy_app_refuses(self):
        world = World()
        world.busy = True
        build(world).set_engine("parakeet")
        self.assertEqual(world.events, [("ui", "show_error", "Finish dictation first",
                                         "Switch engines after the current recording and transcription finish.")])
        self.assertIsNone(world.lifecycle.closed)

    def test_a_closed_gate_refuses(self):
        world = World(closed="updating")
        build(world).set_engine("parakeet")
        self.assertEqual(world.events, [("ui", "show_error", "Could not change engine", "Scrivo is updating.")])
        self.assertEqual(world.lifecycle.closed, "updating")

    def test_parakeet_must_be_downloaded_first(self):
        world = World()
        world.parakeet_ready = False
        build(world).set_engine("parakeet")
        self.assertEqual(world.events, [("ui", "show_error", "Could not change engine",
                                         "Download Parakeet first: Settings → Download Parakeet.")])
        self.assertIsNone(world.lifecycle.closed, "the gate stayed closed")

    def test_nemotron_is_verified_before_the_choice_is_saved(self):
        world = World()
        with patch.object(nemotron_backend, "installation", side_effect=RuntimeError("model missing")):
            build(world).set_engine("nemotron")
        self.assertEqual(world.events, [("ui", "show_error", "Could not change engine", "model missing")])
        self.assertIsNone(world.lifecycle.closed)
        world = World()
        with patch.object(nemotron_backend, "installation") as installation:
            build(world).set_engine("nemotron")
        installation.assert_called_once_with()
        self.assertEqual(world.events, [("persist_engine_and_restart", "nemotron", False)])

    def test_a_failed_restart_reopens_the_gate(self):
        world = World()
        world.persist_failure = "launchd could not restart Scrivo"
        build(world).set_engine("parakeet")
        self.assertEqual(world.events[-1], ("ui", "show_error", "Could not change engine",
                                            "launchd could not restart Scrivo"))
        self.assertIsNone(world.lifecycle.closed)


class OpenDictionaryTests(unittest.TestCase):
    def test_the_dictionary_opens_in_the_text_editor(self):
        world = World()
        with patch.object(dictionary, "ensure_file", return_value=Path("/data/dictionary.txt")) as ensure, \
             patch.object(subprocess, "Popen") as popen:
            build(world).open_dictionary()
        ensure.assert_called_once_with(dictionary.DICTIONARY_PATH)
        popen.assert_called_once_with(["/usr/bin/open", "-t", "/data/dictionary.txt"])
        self.assertEqual(world.events, [])

    def test_an_error_is_shown(self):
        world = World()
        with patch.object(dictionary, "ensure_file", side_effect=OSError("read-only")):
            build(world).open_dictionary()
        self.assertEqual(world.events, [("ui", "show_error", "Could not open the dictionary", "read-only")])


PREFERENCES = {"languages": ["en", "he"], "speed": "normal", "insert_mode": "paste"}


def state_sources():
    """What the Settings state reads besides the run: preferences, the login
    item and the Nemotron install, as fixed values."""
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(patch.object(settings, "load", return_value=dict(PREFERENCES)))
    stack.enter_context(patch.object(login_item, "enabled", return_value=True))
    stack.enter_context(patch.object(nemotron_backend, "is_installed", return_value=False))
    return stack


class SettingsStateTests(unittest.TestCase):
    def state(self, world, *, environ=None):
        with patch.object(settings, "load", return_value=dict(PREFERENCES)), \
             patch.object(login_item, "enabled", return_value=True), \
             patch.object(nemotron_backend, "is_installed", return_value=False), \
             patch.object(dictionary, "load", return_value=["rule"] * 3), \
             patch.dict(os.environ, environ or {}, clear=False):
            for name in ("SCRIVO_LAUNCHER", "SCRIVO_APP_EXECUTABLE"):
                if name not in (environ or {}):
                    os.environ.pop(name, None)
            return build(world).settings_state()

    def test_what_the_window_shows(self):
        from speech_config import ENGINE_CHOICES, WHISPER_LANGUAGES
        world = World()
        world.settings_locked = frozenset({"trigger"})
        self.assertEqual(self.state(world), {
            "settings": PREFERENCES, "trigger": "right-option", "hotkey": "ctrl-opt-d", "locked": {"trigger"},
            "trigger_labels": {name: scrivo.TRIGGER_LABELS[name] for name in settings.TRIGGER_KEYS
                               if name in scrivo.TRIGGERS},
            "engine_labels": dict(ENGINE_CHOICES), "engine": "whisper", "engine_switching": True,
            "nemotron_installed": False, "nemotron_installing": False,
            "parakeet_installed": True, "parakeet_installing": False,
            "fast_available": True, "language_names": WHISPER_LANGUAGES,
            "automatic_languages": ["en", "he"], "login_enabled": True, "login_available": False,
            "dictionary_rules": 3, "data_dir": str(settings.DATA_DIR),
        })

    def test_the_run_decides_what_is_available(self):
        world = World()
        world.adaptive = True
        state = self.state(world, environ={"SCRIVO_LAUNCHER": "app", "SCRIVO_APP_EXECUTABLE": "/A/Scrivo"})
        self.assertEqual((state["engine_switching"], state["fast_available"], state["login_available"]),
                         (False, False, True))
        world = World()
        world.use_nemotron = True
        self.assertFalse(self.state(world)["fast_available"])


class ChangeSettingTests(unittest.TestCase):
    def change(self, world, key, value, *, environ=None, save_error=None):
        saved = []

        def save(changes, path):
            if save_error:
                raise save_error
            saved.append((changes, path))
        with patch.object(settings, "save", save), \
             patch.object(login_item, "enable", lambda executable: world.events.append(("login", executable))), \
             patch.object(login_item, "disable", lambda: world.events.append(("login", None))), \
             patch.dict(os.environ, environ or {}, clear=False):
            for name in ("SCRIVO_LAUNCHER", "SCRIVO_APP_EXECUTABLE"):
                if name not in (environ or {}):
                    os.environ.pop(name, None)
            answer = build(world).change_setting(key, value)
        return answer, saved

    def test_a_flag_locked_setting_is_refused(self):
        world = World()
        world.settings_locked = frozenset({"trigger"})
        self.assertEqual(self.change(world, "trigger", "fn"),
                         ("That setting was chosen with a command-line flag for this run.", []))
        self.assertEqual(world.events, [])

    def test_trigger_and_hotkey_are_saved_and_rebound(self):
        world = World()
        self.assertEqual(self.change(world, "trigger", "fn"), (None, [({"trigger": "fn"}, settings.SETTINGS_PATH)]))
        self.assertEqual(world.events, [("rebind", "fn", None), ("log", "settings: trigger changed")])
        world = World()
        self.assertEqual(self.change(world, "hotkey", "")[0], None)
        self.assertEqual(world.events, [("rebind", None, ""), ("log", "settings: hotkey changed")])

    def test_languages_reach_the_menu_and_history(self):
        world = World()
        self.assertIsNone(self.change(world, "languages", ("en", "fr"))[0])
        self.assertEqual(world.status_ui.automatic_languages, ["en", "fr"])
        self.assertEqual(world.events, [("entries", scrivo.HISTORY_KEEP), ("ui", "refresh_history", []),
                                        ("log", "settings: languages changed")])

    def test_another_setting_is_only_saved(self):
        world = World()
        self.assertEqual(self.change(world, "speed", "fast"), (None, [({"speed": "fast"}, settings.SETTINGS_PATH)]))
        self.assertEqual(world.events, [("log", "settings: speed changed")])

    def test_the_engine_goes_through_the_engine_switch(self):
        world = World()
        self.assertEqual(self.change(world, "engine", "parakeet"), (None, []))
        self.assertEqual(world.events, [("persist_engine_and_restart", "parakeet", False)])

    def test_launch_at_login_needs_the_app(self):
        world = World()
        self.assertEqual(self.change(world, "launch_at_login", True),
                         ("Install Scrivo as an app first (scripts/install-mac.sh).", []))
        app = {"SCRIVO_LAUNCHER": "app", "SCRIVO_APP_EXECUTABLE": "/A/Scrivo"}
        world = World()
        self.assertEqual(self.change(world, "launch_at_login", True, environ=app)[0], None)
        self.assertEqual(world.events, [("login", "/A/Scrivo"), ("log", "settings: launch_at_login changed")])
        world = World()
        self.assertEqual(self.change(world, "launch_at_login", False)[0], None)
        self.assertEqual(world.events[0], ("login", None))

    def test_an_error_is_returned_to_the_window(self):
        world = World()
        self.assertEqual(self.change(world, "speed", "fast", save_error=OSError("disk full"))[0], "disk full")
        self.assertEqual(world.events, [])


class SettingsActionTests(unittest.TestCase):
    def test_buttons(self):
        world = World()
        with patch.object(dictionary, "ensure_file", return_value=Path("/d.txt")), \
             patch.object(subprocess, "Popen") as popen:
            actions = build(world)
            actions.settings_action("open_dictionary")
            actions.settings_action("show_data_folder")
            actions.settings_action("unknown")
        self.assertEqual([call.args[0] for call in popen.call_args_list],
                         [["/usr/bin/open", "-t", "/d.txt"], ["/usr/bin/open", str(scrivo_paths.DATA_DIR)]])

    def test_install_buttons_start_one_install_each(self):
        world = World()
        seen = []
        world.run_installer = lambda argv: (seen.append(actions.settings_state()), (True, "ok"))[1]
        actions = build(world)
        with patch.object(offline_runtime, "offline_requested", return_value=False), state_sources():
            actions.settings_action("install_nemotron")
            actions.settings_action("install_parakeet")
        self.assertEqual([(state["nemotron_installing"], state["parakeet_installing"]) for state in seen],
                         [(True, False), (False, True)])


NEMOTRON_SCRIPT = str(Path(scrivo.__file__).resolve().parent / "scripts" / "setup_nemotron.py")
PARAKEET_SETUP = [sys.executable, str(Path(scrivo.__file__).resolve()), "setup", "--profile", "parakeet"]


class InstallTests(unittest.TestCase):
    def install(self, world, engine, *, offline=False, window=None):
        actions = build(world)
        if window is not None:
            with patch.object(settings_window, "SettingsController", window):
                actions.open_settings()
        with patch.object(offline_runtime, "offline_requested", return_value=offline):
            getattr(actions, f"install_{engine}")()
        return actions

    def test_offline_mode_refuses_the_download(self):
        for engine, message in (
                ("nemotron", "Installing Nemotron downloads about 700 MB; start Scrivo without the offline flags."),
                ("parakeet", "Downloading Parakeet (2.5 GB) needs the network; "
                             "start Scrivo without the offline flags.")):
            with self.subTest(engine=engine):
                world = World()
                self.install(world, engine, offline=True)
                self.assertEqual(world.events, [("ui", "show_error", "Offline mode is on", message)])

    def test_a_finished_install_refreshes_and_says_how_to_use_it(self):
        world = World()
        self.install(world, "nemotron")
        self.assertEqual(world.events, [
            ("thread",), ("log", "● installing Nemotron (about 700 MB) ..."),
            ("run_installer", [sys.executable, NEMOTRON_SCRIPT]),
            ("entries", scrivo.HISTORY_KEEP), ("ui", "refresh_history", []), ("log", "✓ Nemotron installed"),
            ("ui", "show_info", "Nemotron installed", "Choose Speech engine → Nemotron (English streaming) to use it."),
        ])
        world = World()
        self.install(world, "parakeet")
        self.assertEqual(world.events, [
            ("thread",), ("log", "● downloading Parakeet (about 2.5 GB) ..."), ("run_installer", PARAKEET_SETUP),
            ("entries", scrivo.HISTORY_KEEP), ("ui", "refresh_history", []), ("log", "✓ Parakeet downloaded"),
            ("ui", "show_info", "Parakeet downloaded", "Choose Speech engine → Parakeet to use it."),
        ])

    def test_a_failed_install_is_reported_and_an_open_window_refreshed(self):
        detail = "x" * 100
        for engine, logged, title in (("nemotron", "! Nemotron install failed", "Could not install Nemotron"),
                                      ("parakeet", "! Parakeet download failed", "Could not download Parakeet")):
            with self.subTest(engine=engine):
                world = World()
                world.installer_result = (False, detail)
                self.install(world, engine, window=FakeWindow.controller(world))
                self.assertEqual(world.events[-5:], [
                    ("entries", scrivo.HISTORY_KEEP), ("ui", "refresh_history", []), ("ui", "refresh"),
                    ("log", f"{logged} ({'x' * 80})"), ("ui", "show_error", title, detail),
                ])

    def test_the_installing_flag_clears_even_if_the_installer_raises(self):
        for engine, key in (("nemotron", "nemotron_installing"), ("parakeet", "parakeet_installing")):
            with self.subTest(engine=engine):
                world = World()
                world.fail["run_installer"] = OSError("fork failed")
                actions = build(world)
                with patch.object(offline_runtime, "offline_requested", return_value=False), \
                     self.assertRaises(OSError):
                    getattr(actions, f"install_{engine}")()
                with state_sources():
                    self.assertFalse(actions.settings_state()[key])


class FakeWindow:
    """settings_window.SettingsController: alloc().initWithModel_(model)."""

    @classmethod
    def controller(cls, world):
        return types.SimpleNamespace(alloc=lambda: cls(world))

    def __init__(self, world):
        self.world = world
        self.model = None

    def initWithModel_(self, model):
        self.world.events.append(("window", "created"))
        self.model = model
        return self

    def show(self):
        self.world.events.append(("window", "show"))

    def refresh(self):
        pass


class OpenSettingsTests(unittest.TestCase):
    def test_one_window_is_made_and_shown_each_time(self):
        world = World()
        actions = build(world)
        with patch.object(settings_window, "SettingsController", FakeWindow.controller(world)):
            actions.open_settings()
            actions.open_settings()
        self.assertEqual(world.events, [("window", "created"), ("window", "show"), ("window", "show")])

    def test_the_window_reads_and_changes_settings_through_its_model(self):
        world = World()
        windows = []
        controller = FakeWindow.controller(world)
        alloc = controller.alloc
        controller.alloc = lambda: windows.append(alloc()) or windows[-1]
        actions = build(world)
        with patch.object(settings_window, "SettingsController", controller):
            actions.open_settings()
        model = windows[0].model
        with state_sources():
            self.assertEqual(model.state()["trigger"], "right-option")
        with patch.object(settings, "save", lambda changes, path: None):
            self.assertIsNone(model.change("trigger", "fn"))
        self.assertIn(("rebind", "fn", None), world.events)


# -- dictation from the menu ------------------------------------------------------

class FinishNowTests(unittest.TestCase):
    def test_the_gesture_recording_is_finished(self):
        world = World()
        world.recording = True
        build(world).finish_now()
        self.assertEqual(world.events, [("engine.force_finish", True), ("log", "● finished via menu")])

    def test_an_orphan_capture_is_finished_directly(self):
        world = World()
        world.capture_active = True
        build(world).finish_now()
        self.assertEqual(world.events, [("engine.force_finish", True), ("on_finish",),
                                        ("log", "● orphan capture finished via menu")])

    def test_nothing_recording_changes_nothing(self):
        world = World()
        build(world).finish_now()
        self.assertEqual(world.events, [("engine.force_finish", True)])

    def test_after_shutdown_nothing_happens(self):
        world = World(stopping=True)
        world.recording = True
        build(world).finish_now()
        self.assertEqual(world.events, [])


class StartNowTests(unittest.TestCase):
    def test_hands_free_starts(self):
        world = World()
        build(world).start_now()
        self.assertEqual(world.events, [("engine.force_start",),
                                        ("log", "● dictation started via menu (hands-free)")])

    def test_a_refused_start_logs_nothing(self):
        world = World()
        world.start_ok = False
        build(world).start_now()
        self.assertEqual(world.events, [("engine.force_start",)])

    def test_refused_while_the_gate_is_closed_or_stopping(self):
        world = World(closed="updating")
        build(world).start_now()
        self.assertEqual(world.events, [("log", "○ Scrivo is updating — dictation not started")])
        world = World(stopping=True)
        build(world).start_now()
        self.assertEqual(world.events, [])


# -- restart and quit ---------------------------------------------------------------

def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


class RestartTests(unittest.TestCase):
    def test_the_menu_restart_drains_then_restarts(self):
        world = World()
        build(world).restart_when_idle()
        self.assertTrue(wait_for(lambda: ("log", "● restarting Scrivo") in world.events), world.events)
        # The drain runs on its own thread, so the menu's log line may land on either side.
        world.events.remove(("log", "● restart requested from the menu — after the current dictation"))
        self.assertEqual(world.events, [("restart.request",), ("log", "● restarting Scrivo")])
        self.assertEqual(world.lifecycle.closed, "restarting")

    def test_refused_when_the_gate_is_already_closed(self):
        world = World(closed="updating")
        build(world).restart_when_idle()
        self.assertEqual(world.events, [("log", "○ Scrivo is updating — restart not requested")])
        world = World(stopping=True)
        build(world).restart_when_idle()
        self.assertEqual(world.events, [("log", "○ Scrivo is stopping — restart not requested")])

    def test_a_failed_restart_reopens_the_gate_and_says_so(self):
        world = World()
        world.restart_failure = "launchd could not restart Scrivo"
        build(world).restart_when_idle()
        self.assertTrue(wait_for(lambda: ("log", "● restarting Scrivo") in world.events), world.events)
        world.events.remove(("log", "● restart requested from the menu — after the current dictation"))
        self.assertEqual(world.events, [
            ("restart.request",), ("ui", "show_error", "Could not restart Scrivo", "launchd could not restart Scrivo"),
            ("log", "● restarting Scrivo"),
        ])
        self.assertIsNone(world.lifecycle.closed)

    def test_restart_now_reports_a_refusal_or_an_error(self):
        world = World()
        world.restart_accepts = False
        after = []
        build(world).restart_now(after_failure=lambda: after.append(True))
        self.assertEqual(world.events, [("restart.request",), ("ui", "show_error", "Could not restart Scrivo",
                                                               "Scrivo is already shutting down")])
        self.assertEqual(after, [True])
        world = World()
        world.restart_error = OSError("Argument list too long")
        build(world).restart_now()
        self.assertEqual(world.events[-1], ("ui", "show_error", "Could not restart Scrivo", "Argument list too long"))


class QuitTests(unittest.TestCase):
    def test_quit_closes_the_gate_and_stops_once_idle(self):
        world = World()
        build(world).quit()
        self.assertEqual(world.lifecycle.closed, "quitting")
        self.assertTrue(world.shutdown.event.wait(5))


# -- updates ---------------------------------------------------------------------------

UPDATE_LOG = scrivo_paths.DATA_DIR / "logs" / "update.log"
SOURCE = Path(scrivo.__file__).resolve().parent


class ApplyUpdateTests(unittest.TestCase):
    def apply(self, world, outcome, *, app=True):
        calls = []

        def apply_mac(root, python, log_path):
            calls.append((root, python, log_path))
            world.events.append(("apply_mac",))
            return outcome
        with patch.object(updates, "apply_mac", apply_mac), \
             patch.dict(os.environ, {"SCRIVO_LAUNCHER": "app" if app else "terminal"}):
            build(world).apply_update()
        return calls

    def test_a_terminal_run_never_updates_itself(self):
        world = World()
        self.assertEqual(self.apply(world, (True, "", True), app=False), [])
        self.assertEqual(world.events, [])

    def test_a_busy_app_refuses(self):
        world = World()
        world.busy = True
        self.assertEqual(self.apply(world, (True, "", True)), [])
        self.assertEqual(world.events, [("ui", "show_error", "Finish dictation first",
                                         "Update after the current recording and transcription finish.")])

    def test_an_installed_update_drains_then_restarts(self):
        world = World()
        calls = self.apply(world, (True, "", True))
        self.assertEqual(calls, [(SOURCE, sys.executable, UPDATE_LOG)])
        self.assertEqual(world.events, [
            ("log", "● updating Scrivo"), ("thread",), ("apply_mac",),
            ("log", "✓ update installed — restarting once dictation is done"),
            ("wait_until_idle", False, scrivo.UPDATE_DRAIN_DEADLINE_S),
            ("restart.request",), ("log", "● restarting Scrivo"),
        ])
        self.assertEqual(world.lifecycle.closed, "updating")

    def test_the_drain_deadline_restarts_anyway(self):
        world = World()
        world.drained = False
        self.apply(world, (True, "", True))
        self.assertIn(("log", f"! dictation still running after {scrivo.UPDATE_DRAIN_DEADLINE_S:.0f}s — "
                              "restarting into the update anyway"), world.events)
        self.assertEqual(world.events[-2:], [("restart.request",), ("log", "● restarting Scrivo")])

    def test_a_failed_restart_reopens_the_gate(self):
        world = World()
        world.restart_failure = "no launchd"
        self.apply(world, (True, "", True))
        self.assertIsNone(world.lifecycle.closed)

    def test_a_closed_gate_leaves_the_update_for_the_next_start(self):
        world = World(closed="quitting")
        self.apply(world, (True, "", True))
        self.assertEqual(world.events[-1], ("log", "○ Scrivo is quitting; the update applies at the next start"))
        self.assertNotIn(("restart.request",), world.events)

    def test_an_unfinished_or_failed_update_is_reported(self):
        world = World()
        self.apply(world, (False, "pip failed", True))
        self.assertEqual(world.events[-2:], [
            ("log", "! update installed but its setup did not finish (see logs/update.log)"),
            ("ui", "show_error", "Update not finished",
             "pip failed\n\nThe new version is in place but its setup did not finish. "
             "Run scripts/install-mac.sh --update in Terminal to finish it, then "
             "choose Restart Scrivo. Details are in logs/update.log in the data folder."),
        ])
        world = World()
        self.apply(world, (False, "fetch failed", False))
        self.assertEqual(world.events[-2:], [
            ("log", "! update failed (see logs/update.log)"),
            ("ui", "show_error", "Update failed",
             "fetch failed\n\nThe source was not switched, so Scrivo keeps running the "
             "current version. Details are in logs/update.log in the data folder."),
        ])

    def test_one_update_at_a_time(self):
        world = World()
        actions = build(world)
        started = []

        def apply_mac(root, python, log_path):
            started.append(True)
            actions.apply_update()  # a second Update Now while the first runs
            return False, "x", False
        with patch.object(updates, "apply_mac", apply_mac), patch.dict(os.environ, {"SCRIVO_LAUNCHER": "app"}):
            actions.apply_update()
            self.assertEqual(started, [True])
            actions.apply_update()  # the first one finished: allowed again
        self.assertEqual(len(started), 2)


class CheckUpdatesTests(unittest.TestCase):
    def check(self, world, result, *, app=True, answer=False, offline=False):
        world.status_ui.answer = answer
        checked = []

        def check(root, *, offline, started_from):
            checked.append((root, offline, started_from))
            return result
        with patch.object(updates, "check", check), \
             patch.object(offline_runtime, "offline_requested", return_value=offline), \
             patch.object(updates, "apply_mac", return_value=(False, "t", False)), \
             patch.dict(os.environ, {"SCRIVO_LAUNCHER": "app" if app else "terminal"}), \
             patch("AppKit.NSWorkspace") as workspace:
            build(world).check_updates()
        self.assertEqual(checked, [(SOURCE, offline, world.started_from)])
        return workspace

    def test_an_available_update_is_offered_to_the_app(self):
        result = updates.UpdateCheck("available", "", behind=2)
        manual = "run scripts/install-mac.sh --update in Terminal from Scrivo's source folder."
        title, text = updates.describe(result, manual, self_update=True)
        world = World()
        self.check(world, result, answer=True)
        self.assertEqual(world.events[:4], [("thread",), ("log", "update check: available, 2 behind"),
                                            ("ui", "present"), ("direct", "ask", title, text, "Update Now", "Later")])
        self.assertEqual(world.events[4], ("log", "● updating Scrivo"))  # Update Now ran apply_update
        world = World()
        self.check(world, result, answer=False)
        self.assertEqual(len(world.events), 4)

    def test_a_terminal_run_is_told_the_command(self):
        result = updates.UpdateCheck("available", "", behind=1)
        title, text = updates.describe(result, updates.mac_manual_update(SOURCE), self_update=False)
        world = World()
        self.check(world, result, app=False)
        self.assertEqual(world.events[-1], ("direct", "show_info", title, text))

    def test_a_copy_without_git_is_sent_to_the_releases_page(self):
        result = updates.UpdateCheck("not-git", "Not a git checkout.")
        world = World()
        workspace = self.check(world, result, answer=True)
        self.assertEqual(world.events[-1], ("direct", "ask", "Download the latest release", "Not a git checkout.",
                                            "Open Releases Page", "Close"))
        opened = workspace.sharedWorkspace.return_value.openURL_.call_args.args[0]
        self.assertEqual(str(opened.absoluteString()), updates.RELEASES_URL)

    def test_a_failure_is_an_error_and_offline_says_so(self):
        result = updates.UpdateCheck("failed", "git fetch failed")
        title, text = updates.describe(result, "run scripts/install-mac.sh --update in Terminal from "
                                               "Scrivo's source folder.", self_update=True)
        world = World()
        self.check(world, result)
        self.assertEqual(world.events[-1], ("direct", "show_error", title, text))
        world = World()
        self.check(world, updates.UpdateCheck("offline", "Offline mode is on."), offline=True)
        self.assertEqual(world.events[1], ("log", "update check: offline"))
        self.assertEqual(world.events[-1][:2], ("direct", "show_info"))

    # 2026-10-10: a backup job's git merge moved the running checkout from
    # c2e7f0f to 3bcf356, and Check for Updates said "up to date" while this
    # process still ran c2e7f0f.
    APP_MANUAL = "run scripts/install-mac.sh --update in Terminal from Scrivo's source folder."

    def stale(self, state="current", *, packages_changed=False, detail=""):
        return updates.UpdateCheck(state, detail, version="3bcf356",
                                   stale=updates.StaleRun("c2e7f0f", "3bcf356", packages_changed))

    def test_a_stale_run_with_the_same_packages_offers_the_drained_restart(self):
        result = self.stale()
        title, text = updates.describe(result, self.APP_MANUAL, self_update=True)
        world = World()
        self.check(world, result, answer=True)
        self.assertEqual(world.events[:4], [
            ("thread",), ("log", "update check: current; running c2e7f0f, files 3bcf356"),
            ("ui", "present"), ("direct", "ask", title, text, "Restart Now", "Later")])
        # The menu's Restart: the gate closes and the dictation in flight finishes first.
        self.assertTrue(wait_for(lambda: ("log", "● restarting Scrivo") in world.events), world.events)
        self.assertIn(("log", "● restart requested from the menu — after the current dictation"), world.events)
        self.assertEqual(world.lifecycle.closed, "restarting")
        world = World()
        self.check(world, result, answer=False)
        self.assertEqual(len(world.events), 4)
        self.assertIsNone(world.lifecycle.closed)

    def test_a_stale_run_with_new_packages_is_offered_the_installer_not_a_restart(self):
        result = self.stale(packages_changed=True)
        title, text = updates.describe(result, self.APP_MANUAL, self_update=True)
        world = World()
        self.check(world, result, answer=True)
        self.assertEqual(world.events[:5], [
            ("thread",), ("log", "update check: current; running c2e7f0f, files 3bcf356, packages changed"),
            ("ui", "present"), ("direct", "ask", title, text, "Update Now", "Later"),
            ("log", "● updating Scrivo")])  # apply_update: install-mac.sh --update, then the drained restart
        self.assertNotIn(("restart.request",), world.events[:5])
        world = World()
        self.check(world, self.stale("offline", packages_changed=True, detail="Offline mode is on."), offline=True)
        title, text = updates.describe(self.stale("offline", packages_changed=True, detail="Offline mode is on."),
                                       self.APP_MANUAL, self_update=True)
        self.assertEqual(world.events[-1], ("direct", "show_info", title, text))

    def test_a_stale_terminal_run_is_told_what_to_do(self):
        for packages_changed in (False, True):
            result = self.stale(packages_changed=packages_changed)
            title, text = updates.describe(result, updates.mac_manual_update(SOURCE), self_update=False)
            world = World()
            self.check(world, result, app=False, answer=True)
            self.assertEqual(world.events[-1], ("direct", "show_info", title, text))
            self.assertNotIn(("restart.request",), world.events)
            self.assertIsNone(world.lifecycle.closed)


# -- the menu bar ------------------------------------------------------------------

class MenuCallbacksTests(unittest.TestCase):
    ITEMS = {"quit": "quit", "restart": "restart_when_idle", "copy": "copy", "retry": "retry", "save": "save",
             "delete": "delete", "clear": "clear", "correct": "correct", "enroll": "enroll", "revoke": "revoke",
             "correct_as_is": "correct_as_is", "no_speech": "no_speech",
             "skip_learning_review": "skip_learning_review", "finish_now": "finish_now",
             "start_now": "start_now", "set_language": "set_language", "set_engine": "set_engine",
             "add_rules": "add_rules", "open_dictionary": "open_dictionary", "open_settings": "open_settings",
             "check_updates": "check_updates"}

    def wired(self, world):
        controller = build(world)
        called = []
        for item, method in self.ITEMS.items():
            setattr(controller, method, lambda *args, _item=item, **kwargs: called.append((_item, args, kwargs)))
        return controller.menu_callbacks(), called

    def test_each_item_runs_its_action_with_its_arguments(self):
        callbacks, called = self.wired(World())
        self.assertEqual(set(callbacks), set(self.ITEMS))
        for item in self.ITEMS:
            callbacks[item]("row1", expected_revision=2)
        self.assertEqual(called, [(item, ("row1",), {"expected_revision": 2}) for item in self.ITEMS])

    def test_once_stopping_only_quit_runs(self):
        world = World()
        callbacks, called = self.wired(world)
        world.shutdown.request()
        self.assertEqual({item: callbacks[item]("row1") for item in self.ITEMS if item != "quit"},
                         {item: None for item in self.ITEMS if item != "quit"})
        callbacks["quit"]()
        self.assertEqual(called, [("quit", (), {})])


if __name__ == "__main__":
    unittest.main()
