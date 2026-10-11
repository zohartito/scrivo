"""The menu-bar actions of the Mac app (History, settings, lifecycle).

Each test drives one action with fakes (no AppKit main loop, no clipboard,
no model, no mic, no threads: background work runs inline and is recorded
as a "thread" event) and checks everything it does: History and learning
calls, deliveries, queued jobs, UI calls and log lines.

The expected values were recorded from the actions as closures inside
sotto.run(), before they moved into AppController; the class reproduces
them exactly.
"""
from __future__ import annotations

import ast
import os
import queue
import subprocess
import sys
import threading
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
import sotto
import sotto_paths
from app_controller import AppController

CONFIG = "speech-config"


class World:
    """The collaborators an action reaches, as fakes that record each call
    in ``events`` (in order)."""

    def __init__(self, *, rows=(), adaptive=None, closed=None, stopping=False):
        self.events: list = []
        self.rows = {row["id"]: dict(row) for row in rows}
        self.shutdown = sotto.ShutdownBoundary()
        self.gate = sotto.CaptureGate()
        self.busy = False
        self.lifecycle = sotto.Lifecycle(self.shutdown, self.gate, end_recording=lambda: None,
                                         busy=lambda: self.busy)
        if closed:
            self.assert_true(self.lifecycle.close(closed))
        if stopping:
            self.shutdown.request()
        self.jobs: queue.Queue = queue.Queue()
        self.pending = sotto.PendingDeliveries()
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
        self.restart = types.SimpleNamespace(request=lambda on_failure=None: True)
        self.parakeet_ready = True
        self.persist_failure = None   # the restart's on_failure message
        self.installer_result = (True, "done")

    @staticmethod
    def assert_true(value):
        assert value

    # -- helpers run() hands over --------------------------------------------
    def ui_call(self, method, *args):
        self.events.append(("ui", method.__name__, *args))

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
        return folder / "sotto-20261010-120000.wav"

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


def gate_open(gate) -> bool:
    with gate.starting() as allowed:
        return allowed


class Recorder:
    """status_ui: every method is a named no-op the World records through ui_call."""

    def __init__(self, events):
        self._events = events

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def method(*args):
            self._events.append(("direct", name, *args))
        method.__name__ = name
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
        current_speech_config=world.current_speech_config, log=world.log, history_keep=sotto.HISTORY_KEEP,
        overlay_learning_state=sotto.overlay_learning_state,
        active_learning_history_ids=world.active_learning_history_ids,
        nonadaptive_correct=world.nonadaptive_correct, nonadaptive_delete=world.nonadaptive_delete,
        nonadaptive_clear=world.nonadaptive_clear, nonadaptive_revoke_learning=world.nonadaptive_revoke_learning,
        save_audio_copy=world.save_audio_copy, start_thread=world.start_thread)


# -- History ------------------------------------------------------------------

class RefreshHistoryTests(unittest.TestCase):
    def test_rows_get_the_learning_overlay(self):
        world = World(rows=[ROW, {"id": "active-row", "text": "x", "ts": 2.0}])
        build(world).refresh_history()
        self.assertEqual(world.events, [
            ("entries", sotto.HISTORY_KEEP),
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
                    ("entries", sotto.HISTORY_KEEP),
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
            ("log", "○ Sotto is updating — Retry refused"),
            ("ui", "show_error", "Not retried", "Sotto is updating. Retry once it is running again."),
        ])

    def test_retry_is_refused_while_stopping(self):
        world = World(stopping=True)
        build(world).retry("row1")
        self.assertTrue(world.jobs.empty())
        self.assertEqual(world.events[0], ("log", "○ Sotto is stopping — Retry refused"))


class SaveAudioTests(unittest.TestCase):
    def test_save_copies_the_audio_to_the_desktop(self):
        world = World(rows=[ROW])
        build(world).save("row1")
        self.assertEqual(world.events, [
            ("thread",),
            ("save_audio_copy", Path("/history/audio/row1.wav"), Path.home() / "Desktop", ROW["ts"]),
            ("log", "✓ saved audio to Desktop as sotto-20261010-120000.wav"),
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
            ("entries", sotto.HISTORY_KEEP), ("ui", "refresh_history", [dict(ROW)]),
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
                                        ("entries", sotto.HISTORY_KEEP), ("ui", "refresh_history", [])])
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
            ("entries", sotto.HISTORY_KEEP), ("ui", "refresh_history", [dict(ROW)]),
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
                                        ("entries", sotto.HISTORY_KEEP), ("ui", "refresh_history", [])])

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
            ("thread",), ("enroll", "row1"), ("entries", sotto.HISTORY_KEEP), ("ui", "refresh_history", []),
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
            ("thread",), ("nonadaptive_revoke_learning", "row1"), ("entries", sotto.HISTORY_KEEP),
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
                self.assertEqual(world.events, [("thread",), review, ("entries", sotto.HISTORY_KEEP),
                                                ("ui", "set_adaptive_stage", "adaptive: collecting challenger"),
                                                ("ui", "refresh_history", [])])


# -- Settings: the closures in sotto.run(), compiled against the World ---------

SETTINGS_CLOSURES = ("action_set_language", "action_set_engine", "action_open_dictionary", "SettingsModel",
                     "install_nemotron_in_background", "install_parakeet_in_background",
                     "action_open_settings")


def _run_definitions():
    source = Path(sotto.__file__).read_text(encoding="utf-8")
    run = next(node for node in ast.parse(source).body
               if isinstance(node, ast.FunctionDef) and node.name == "run")
    found = {}

    def visit(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                found.setdefault(node.name, node)
            elif isinstance(node, (ast.If, ast.With, ast.Try)):
                visit(node.body)
                visit(getattr(node, "orelse", []))
    visit(run.body)
    return found


def build_settings(world: World):
    """The Settings actions as run() defines them, closed over the World's fakes."""
    class SyncThread:
        def __init__(self, target, daemon):
            assert daemon
            self.target = target

        def start(self):
            world.start_thread(self.target)
    namespace = {
        "adaptive": world.adaptive, "use_nemotron": world.use_nemotron, "use_parakeet": world.use_parakeet,
        "engine_switching": world.engine_switching, "active_engine": world.active_engine,
        "settings_locked": world.settings_locked, "binding": world.binding, "rebind": world.rebind,
        "restart": world.restart, "lifecycle": world.lifecycle, "status_ui": world.status_ui,
        "ui_call": world.ui_call, "log": world.log, "save_language_mode": world.save_language_mode,
        "language_lock": world.language_lock, "language_state": world.language_state,
        "parakeet_cached": world.parakeet_cached, "persist_engine_and_restart": world.persist_engine_and_restart,
        "run_installer": world.run_installer, "refresh_history": build(world).refresh_history,
        "TRIGGERS": sotto.TRIGGERS, "TRIGGER_LABELS": sotto.TRIGGER_LABELS, "app_mode": sotto.app_mode,
        "os": os, "sys": sys, "Path": Path, "threading": types.SimpleNamespace(Thread=SyncThread),
        "__file__": sotto.__file__,
        "settings_view": {"controller": None, "installing": False, "parakeet_installing": False},
    }
    definitions = _run_definitions()
    module = ast.Module(body=[definitions[name] for name in SETTINGS_CLOSURES], type_ignores=[])
    exec(compile(module, sotto.__file__, "exec"), namespace)
    model = namespace["SettingsModel"]
    return types.SimpleNamespace(
        set_language=namespace["action_set_language"], set_engine=namespace["action_set_engine"],
        open_dictionary=namespace["action_open_dictionary"], settings_state=lambda: model().state(),
        change_setting=lambda key, value: model().change(key, value),
        settings_action=lambda name: model().action(name),
        install_nemotron=namespace["install_nemotron_in_background"],
        install_parakeet=namespace["install_parakeet_in_background"],
        open_settings=namespace["action_open_settings"])


# -- Settings -------------------------------------------------------------------

class LanguageTests(unittest.TestCase):
    def test_the_choice_is_saved_shown_and_used_by_the_next_recording(self):
        world = World()
        build_settings(world).set_language("en")
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
                build_settings(world).set_language("en")
                self.assertEqual((world.events, world.language_state), ([], {"mode": "auto"}))

    def test_a_save_error_is_shown_and_changes_nothing(self):
        world = World()
        world.fail["save_language_mode"] = OSError("read-only")
        build_settings(world).set_language("he")
        self.assertEqual(world.language_state, {"mode": "auto"})
        self.assertEqual(world.events[-1], ("ui", "show_error", "Could not change language", "read-only"))


class EngineTests(unittest.TestCase):
    def test_switching_closes_the_gate_then_persists_and_restarts(self):
        world = World()
        build_settings(world).set_engine("parakeet")
        self.assertEqual(world.events, [("persist_engine_and_restart", "parakeet", False)])
        self.assertEqual(world.lifecycle.closed, "switching engines")

    def test_nothing_happens_for_the_current_engine_adaptive_or_a_locked_engine(self):
        for flag, value in (("active_engine", "parakeet"), ("adaptive", True), ("engine_switching", False)):
            with self.subTest(flag=flag):
                world = World()
                setattr(world, flag, value)
                build_settings(world).set_engine("parakeet")
                self.assertEqual(world.events, [])
                self.assertIsNone(world.lifecycle.closed)

    def test_a_busy_app_refuses(self):
        world = World()
        world.busy = True
        build_settings(world).set_engine("parakeet")
        self.assertEqual(world.events, [("ui", "show_error", "Finish dictation first",
                                         "Switch engines after the current recording and transcription finish.")])
        self.assertIsNone(world.lifecycle.closed)

    def test_a_closed_gate_refuses(self):
        world = World(closed="updating")
        build_settings(world).set_engine("parakeet")
        self.assertEqual(world.events, [("ui", "show_error", "Could not change engine", "Sotto is updating.")])
        self.assertEqual(world.lifecycle.closed, "updating")

    def test_parakeet_must_be_downloaded_first(self):
        world = World()
        world.parakeet_ready = False
        build_settings(world).set_engine("parakeet")
        self.assertEqual(world.events, [("ui", "show_error", "Could not change engine",
                                         "Download Parakeet first: Settings → Download Parakeet.")])
        self.assertIsNone(world.lifecycle.closed, "the gate stayed closed")

    def test_nemotron_is_verified_before_the_choice_is_saved(self):
        world = World()
        with patch.object(nemotron_backend, "installation", side_effect=RuntimeError("model missing")):
            build_settings(world).set_engine("nemotron")
        self.assertEqual(world.events, [("ui", "show_error", "Could not change engine", "model missing")])
        self.assertIsNone(world.lifecycle.closed)
        world = World()
        with patch.object(nemotron_backend, "installation") as installation:
            build_settings(world).set_engine("nemotron")
        installation.assert_called_once_with()
        self.assertEqual(world.events, [("persist_engine_and_restart", "nemotron", False)])

    def test_a_failed_restart_reopens_the_gate(self):
        world = World()
        world.persist_failure = "launchd could not restart Sotto"
        build_settings(world).set_engine("parakeet")
        self.assertEqual(world.events[-1], ("ui", "show_error", "Could not change engine",
                                            "launchd could not restart Sotto"))
        self.assertIsNone(world.lifecycle.closed)


class OpenDictionaryTests(unittest.TestCase):
    def test_the_dictionary_opens_in_the_text_editor(self):
        world = World()
        with patch.object(dictionary, "ensure_file", return_value=Path("/data/dictionary.txt")) as ensure, \
             patch.object(subprocess, "Popen") as popen:
            build_settings(world).open_dictionary()
        ensure.assert_called_once_with(dictionary.DICTIONARY_PATH)
        popen.assert_called_once_with(["/usr/bin/open", "-t", "/data/dictionary.txt"])
        self.assertEqual(world.events, [])

    def test_an_error_is_shown(self):
        world = World()
        with patch.object(dictionary, "ensure_file", side_effect=OSError("read-only")):
            build_settings(world).open_dictionary()
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
            for name in ("SOTTO_LAUNCHER", "SOTTO_APP_EXECUTABLE"):
                if name not in (environ or {}):
                    os.environ.pop(name, None)
            return build_settings(world).settings_state()

    def test_what_the_window_shows(self):
        from speech_config import ENGINE_CHOICES, WHISPER_LANGUAGES
        world = World()
        world.settings_locked = frozenset({"trigger"})
        self.assertEqual(self.state(world), {
            "settings": PREFERENCES, "trigger": "right-option", "hotkey": "ctrl-opt-d", "locked": {"trigger"},
            "trigger_labels": {name: sotto.TRIGGER_LABELS[name] for name in settings.TRIGGER_KEYS
                               if name in sotto.TRIGGERS},
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
        state = self.state(world, environ={"SOTTO_LAUNCHER": "app", "SOTTO_APP_EXECUTABLE": "/A/Sotto"})
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
            for name in ("SOTTO_LAUNCHER", "SOTTO_APP_EXECUTABLE"):
                if name not in (environ or {}):
                    os.environ.pop(name, None)
            answer = build_settings(world).change_setting(key, value)
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
        self.assertEqual(world.events, [("entries", sotto.HISTORY_KEEP), ("ui", "refresh_history", []),
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
                         ("Install Sotto as an app first (scripts/install-mac.sh).", []))
        app = {"SOTTO_LAUNCHER": "app", "SOTTO_APP_EXECUTABLE": "/A/Sotto"}
        world = World()
        self.assertEqual(self.change(world, "launch_at_login", True, environ=app)[0], None)
        self.assertEqual(world.events, [("login", "/A/Sotto"), ("log", "settings: launch_at_login changed")])
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
            actions = build_settings(world)
            actions.settings_action("open_dictionary")
            actions.settings_action("show_data_folder")
            actions.settings_action("unknown")
        self.assertEqual([call.args[0] for call in popen.call_args_list],
                         [["/usr/bin/open", "-t", "/d.txt"], ["/usr/bin/open", str(sotto_paths.DATA_DIR)]])

    def test_install_buttons_start_one_install_each(self):
        world = World()
        seen = []
        world.run_installer = lambda argv: (seen.append(actions.settings_state()), (True, "ok"))[1]
        actions = build_settings(world)
        with patch.object(offline_runtime, "offline_requested", return_value=False), state_sources():
            actions.settings_action("install_nemotron")
            actions.settings_action("install_parakeet")
        self.assertEqual([(state["nemotron_installing"], state["parakeet_installing"]) for state in seen],
                         [(True, False), (False, True)])


NEMOTRON_SCRIPT = str(Path(sotto.__file__).resolve().parent / "scripts" / "setup_nemotron.py")
PARAKEET_SETUP = [sys.executable, str(Path(sotto.__file__).resolve()), "setup", "--profile", "parakeet"]


class InstallTests(unittest.TestCase):
    def install(self, world, engine, *, offline=False, window=None):
        actions = build_settings(world)
        if window is not None:
            with patch.object(settings_window, "SettingsController", window):
                actions.open_settings()
        with patch.object(offline_runtime, "offline_requested", return_value=offline):
            getattr(actions, f"install_{engine}")()
        return actions

    def test_offline_mode_refuses_the_download(self):
        for engine, message in (
                ("nemotron", "Installing Nemotron downloads about 700 MB; start Sotto without the offline flags."),
                ("parakeet", "Downloading Parakeet (2.5 GB) needs the network; "
                             "start Sotto without the offline flags.")):
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
            ("entries", sotto.HISTORY_KEEP), ("ui", "refresh_history", []), ("log", "✓ Nemotron installed"),
            ("ui", "show_info", "Nemotron installed", "Choose Speech engine → Nemotron (English streaming) to use it."),
        ])
        world = World()
        self.install(world, "parakeet")
        self.assertEqual(world.events, [
            ("thread",), ("log", "● downloading Parakeet (about 2.5 GB) ..."), ("run_installer", PARAKEET_SETUP),
            ("entries", sotto.HISTORY_KEEP), ("ui", "refresh_history", []), ("log", "✓ Parakeet downloaded"),
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
                    ("entries", sotto.HISTORY_KEEP), ("ui", "refresh_history", []), ("ui", "refresh"),
                    ("log", f"{logged} ({'x' * 80})"), ("ui", "show_error", title, detail),
                ])

    def test_the_installing_flag_clears_even_if_the_installer_raises(self):
        for engine, key in (("nemotron", "nemotron_installing"), ("parakeet", "parakeet_installing")):
            with self.subTest(engine=engine):
                world = World()
                world.fail["run_installer"] = OSError("fork failed")
                actions = build_settings(world)
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
        actions = build_settings(world)
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
        actions = build_settings(world)
        with patch.object(settings_window, "SettingsController", controller):
            actions.open_settings()
        model = windows[0].model
        with state_sources():
            self.assertEqual(model.state()["trigger"], "right-option")
        with patch.object(settings, "save", lambda changes, path: None):
            self.assertIsNone(model.change("trigger", "fn"))
        self.assertIn(("rebind", "fn", None), world.events)


if __name__ == "__main__":
    unittest.main()
