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

import queue
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

if sys.platform != "darwin":
    raise unittest.SkipTest("the Mac menu actions — macOS-only")

import dictionary
import sotto
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
        self.lifecycle = sotto.Lifecycle(self.shutdown, self.gate, end_recording=lambda: None,
                                         busy=lambda: False)
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


if __name__ == "__main__":
    unittest.main()
