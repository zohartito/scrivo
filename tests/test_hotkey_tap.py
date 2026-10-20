"""Characterization of the event tap through the real ``scrivo.run()``.

Restructure step 6 pins what the tap decides before it moves out of run():
these tests build the whole app with fakes (no menu bar, no event tap, no
microphone, no model), capture the callback run() hands to
``CGEventTapCreate`` and the poller thread it starts, then feed key events and
poll ticks and watch what reaches the gesture engine, the log and the menu bar.

The gesture engine is a recording fake with a scripted ``snapshot()``: the
tap's decisions are which engine calls it makes, and when. Quartz keeps its
real constants; only the functions that touch the window server are faked.
The poller's one-second sleep is gated, so a test steps it one poll at a time.
"""
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import history
import learning
import scrivo
from speech_config import resolve_speech_config

if sys.platform == "darwin":
    import Quartz
    from tests.test_nemotron import FakeRuntime

REAL_THREAD = threading.Thread
REAL_SLEEP = time.sleep
TAP = object()        # what the fake CGEventTapCreate returns
RIGHT_OPTION = 61     # keycode
RIGHT_OPTION_BIT = 0x0040
LEFT_OPTION_BIT = 0x0020
D, A, S, F13 = 2, 0, 1, 105
WAIT_S = 5.0


class FakeEngine:
    """Stands in for GestureEngine: records every call; ``recording`` and
    ``hands_free`` are what snapshot() reports."""

    def __init__(self, on_start, on_finish, on_discard, on_hands_free=None):
        self.calls = []
        self.recording = False
        self.hands_free = False
        self.start_ok = True

    def snapshot(self):
        return self.recording, self.hands_free

    def pressed(self):
        self.calls.append("pressed")

    def released(self):
        self.calls.append("released")

    def chorded(self):
        self.calls.append("chorded")

    def force_start(self):
        self.calls.append("force_start")
        if self.start_ok and not self.recording:
            self.recording = True
            return True
        return False

    def force_finish(self, finish=None, if_epoch=None):
        self.calls.append("force_finish")
        if self.recording:
            self.recording = False
            return True
        return False

    def force_reset(self):
        self.calls.append("force_reset")

    def epoch(self):
        return 0


class FakeTapOS:
    """The window-server side of the tap: CGEventTapCreate/Enable/IsEnabled,
    CGEventSourceFlagsState (``flags_state``) and the event accessors."""

    def __init__(self, *, create_ok=True):
        self.create_ok = create_ok
        self.created = []
        self.enables = []          # (enable, thread) per CGEventTapEnable call
        self.enabled = False
        self.stay_disabled = False  # CGEventTapEnable(tap, True) does not stick
        self.flags_state = 0
        self.sources = []
        self.callback = None

    def tap_create(self, tap, place, options, mask, callback, refcon):
        self.created.append((tap, place, options, mask, refcon))
        self.callback = callback
        return TAP if self.create_ok else None

    def tap_enable(self, tap, enable):
        assert tap is TAP, tap
        self.enables.append((bool(enable), threading.current_thread()))
        self.enabled = bool(enable) and not self.stay_disabled

    def tap_is_enabled(self, tap):
        assert tap is TAP, tap
        return self.enabled

    def flags_state_of(self, state):
        assert state == Quartz.kCGEventSourceStateCombinedSessionState, state
        return self.flags_state

    @staticmethod
    def integer_field(event, field):
        if field == Quartz.kCGKeyboardEventKeycode:
            return event["keycode"]
        if field == Quartz.kCGKeyboardEventAutorepeat:
            return event["autorepeat"]
        if field == Quartz.kCGEventSourceUserData:
            return event["user_data"]
        raise AssertionError(f"unexpected field {field}")

    @staticmethod
    def flags(event):
        return event["flags"]

    def add_source(self, loop, source, mode):
        self.sources.append((source, mode))

    def replacements(self):
        return {
            "Quartz.CGEventTapCreate": self.tap_create,
            "Quartz.CGEventTapEnable": self.tap_enable,
            "Quartz.CGEventTapIsEnabled": self.tap_is_enabled,
            "Quartz.CGEventSourceFlagsState": self.flags_state_of,
            "Quartz.CGEventGetIntegerValueField": self.integer_field,
            "Quartz.CGEventGetFlags": self.flags,
            "Quartz.CFMachPortCreateRunLoopSource": lambda allocator, tap, order: ("source", tap, order),
            "Quartz.CFRunLoopAddSource": self.add_source,
        }


class PollerGate:
    """Runs the poller run() starts on a thread the test steps: its
    time.sleep parks until tick() lets one poll through."""

    def __init__(self):
        self.target = None
        self.watchdog = None                # tap_watchdog's target: the test calls watchdog_pass() itself
        self.thread = None
        self.sleeps = []
        self._parked = threading.Semaphore(0)
        self._go = threading.Semaphore(0)

    def sleep(self, seconds):
        if threading.current_thread() is not self.thread:
            return REAL_SLEEP(seconds)
        self.sleeps.append(seconds)
        self._parked.release()
        self._go.acquire()

    def start(self):
        self.thread = REAL_THREAD(target=self.target, daemon=True, name="test-resync-poller")
        self.thread.start()
        assert self._parked.acquire(timeout=WAIT_S), "the poller never slept"

    def tick(self):
        """One poll: wake the poller and wait until it sleeps again."""
        self._go.release()
        assert self._parked.acquire(timeout=WAIT_S), "the poller did not come back to sleep"

    def stop(self):
        if self.thread is not None and self.thread.is_alive():
            self._go.release()
            self.thread.join(WAIT_S)


def event(keycode, flags=0, *, autorepeat=0, user_data=0):
    return {"keycode": keycode, "flags": flags, "autorepeat": autorepeat, "user_data": user_data}


class App:
    """What a test sees of one run(): the tap callback, the engine, the log,
    the menu bar mock, the poller and the collaborators run() wired."""

    def __init__(self, tap_os, gate):
        self.tap_os = tap_os
        self.gate = gate
        self.logs = []
        self.engines = []
        self.boundaries = []
        self.captures = []
        self.delivery_kwargs = {}
        self.controller_kwargs = {}
        self.status = MagicMock()
        self.called_after = []     # names of what went through AppHelper.callAfter

    def call_after(self, fn, *args):
        self.called_after.append(getattr(fn, "__name__", repr(fn)))
        fn(*args)

    @property
    def engine(self):
        return self.engines[0]

    def send(self, event_type, payload):
        returned = self.tap_os.callback(None, event_type, payload, None)
        assert returned is payload, "the tap is listen-only: it hands every event back"
        return returned

    def flags_changed(self, keycode, flags, **kw):
        return self.send(Quartz.kCGEventFlagsChanged, event(keycode, flags, **kw))

    def key_down(self, keycode, flags=0, **kw):
        return self.send(Quartz.kCGEventKeyDown, event(keycode, flags, **kw))

    def key_up(self, keycode, flags=0, **kw):
        return self.send(Quartz.kCGEventKeyUp, event(keycode, flags, **kw))

    def press_trigger(self, device_bit=RIGHT_OPTION_BIT, keycode=RIGHT_OPTION,
                      mask=None):
        mask = Quartz.kCGEventFlagMaskAlternate if mask is None else mask
        self.flags_changed(keycode, mask | device_bit)

    def release_trigger(self, keycode=RIGHT_OPTION):
        self.flags_changed(keycode, 0)

    def hotkey_flags(self):
        return Quartz.kCGEventFlagMaskControl | Quartz.kCGEventFlagMaskAlternate

    def keydowns(self):
        return self.delivery_kwargs["keydowns"]()

    def logged(self, prefix):
        return [line for line in self.logs if line.startswith(prefix)]

    def tap_health(self):
        return [call.args[0] for call in self.status.set_tap_health.call_args_list]


def run_app(test, drive, *, trigger="right-option", hotkey_spec="ctrl-opt-d", create_ok=True):
    """scrivo.run() with every OS boundary faked; drive(app) runs as the main
    loop, then shutdown is requested and run() returns. Returns the App."""
    tap_os, gate = FakeTapOS(create_ok=create_ok), PollerGate()
    app = test.app = App(tap_os, gate)  # test.app: also when run() raises
    status = app.status
    status.set_history_callbacks.side_effect = lambda callbacks: None

    started = []
    ns_app = MagicMock()

    class InterceptThread(REAL_THREAD):
        def start(self):
            name = getattr(self._target, "__name__", "")
            if name == "resync_poller":
                gate.target = self._target  # the test runs it, one poll at a time
                return
            if name == "tap_watchdog":
                gate.watchdog = self._target  # one pass at a time, from the test's own thread
                return
            started.append(name)
            super().start()

    class RecordingEngine(FakeEngine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            app.engines.append(self)

    class RecordingBoundary(scrivo.ShutdownBoundary):
        def __init__(self):
            super().__init__()
            app.boundaries.append(self)

    class Capture(scrivo.CaptureService):
        def __init__(self):
            super().__init__()
            self.ticks = 0
            app.captures.append(self)

        def _start_engine(self):
            self._starting = False

        def tick(self):
            self.ticks += 1

    def runtime():
        fake = FakeRuntime()
        fake.close = lambda: None
        return fake

    real_delivery, real_controller = scrivo.DeliveryQueue, scrivo.AppController

    def delivery_queue(**kwargs):
        app.delivery_kwargs.update(kwargs)
        return real_delivery(**kwargs)

    def app_controller(**kwargs):
        app.controller_kwargs.update(kwargs)
        return real_controller(**kwargs)

    def run_loop():
        try:
            gate.start()
            drive(app)
        finally:
            app.boundaries[0].request()
            gate.stop()

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory, ExitStack() as stack:
        root = Path(directory)
        real_store, real_learning = history.HistoryStore, learning.LearningStore
        replacements = {
            **tap_os.replacements(),
            "scrivo.CaptureService": Capture,
            "scrivo.ShutdownBoundary": RecordingBoundary,
            "scrivo.GestureEngine": RecordingEngine,
            "scrivo.DeliveryQueue": delivery_queue,
            "scrivo.AppController": app_controller,
            "scrivo.log": app.logs.append,
            "scrivo.microphone_permission": lambda **_kw: True,
            "scrivo.inject": lambda text, **_kw: test.fail(f"pasted {text!r}"),
            "nemotron_backend.NemotronRuntime": runtime,
            "history.HistoryStore": lambda **kw: real_store(root, **kw),
            "learning.LearningStore": lambda **kw: real_learning(root, **kw),
            "ui.init_app": lambda: status,
            "ui.run_loop": run_loop,
            "ApplicationServices.AXIsProcessTrustedWithOptions": lambda _: True,
            "PyObjCTools.AppHelper.callAfter": app.call_after,
            "AppKit.NSApp": ns_app,
            "threading.Thread": InterceptThread,
            "time.sleep": gate.sleep,
        }
        for name, replacement in replacements.items():
            stack.enter_context(patch(name, replacement))
        try:
            scrivo.run(trigger, speech_config=resolve_speech_config("nemotron-en"), hotkey_spec=hotkey_spec)
        finally:
            for boundary in app.boundaries:
                boundary.request()  # also when run() exits early: let its worker thread end
            # The shutdown watcher tears down on its own thread (it disables
            # the tap, then terminates NSApp last): let it finish under the
            # fakes, never against the real window server.
            deadline = time.monotonic() + WAIT_S
            while ("shutdown_watcher" in started and not ns_app.terminate_.called
                   and time.monotonic() < deadline):
                REAL_SLEEP(0.01)
    return app


@unittest.skipUnless(sys.platform == "darwin", "drives scrivo.run(), which needs PyObjC; macOS-only")
class TapSetupTests(unittest.TestCase):
    def test_run_records_the_commit_it_started_from_once(self):
        # Check for Updates compares the folder's HEAD with it (2026-10-10:
        # a git merge under the running app was reported as "up to date").
        started = "c2e7f0f" + "0" * 33
        with patch("updates.started_from", return_value=started) as started_from:
            app = run_app(self, lambda app: None)
        started_from.assert_called_once_with(Path(scrivo.__file__).resolve().parent)
        self.assertEqual(app.controller_kwargs["started_from"], started)

    def test_a_listen_only_session_tap_on_flags_and_keys_is_added_to_the_run_loop_and_enabled(self):
        seen = {}

        def drive(app):
            seen["created"] = list(app.tap_os.created)
            seen["enables"] = [enable for enable, _ in app.tap_os.enables]
            seen["sources"] = list(app.tap_os.sources)

        app = run_app(self, drive)
        mask = (Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged) | Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown)
                | Quartz.CGEventMaskBit(Quartz.kCGEventKeyUp))
        self.assertEqual(seen["created"], [(Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap,
                                            Quartz.kCGEventTapOptionListenOnly, mask, None)])
        self.assertEqual(seen["sources"], [(("source", TAP, 0), Quartz.kCFRunLoopCommonModes)])
        self.assertEqual(seen["enables"], [True])
        self.assertIsNotNone(app.gate.target, "run() starts the resync poller")
        self.assertEqual(getattr(app.gate.watchdog, "__name__", None), "tap_watchdog",
                         "run() starts the tap watchdog on a thread of its own")
        listening = app.logged("listening on")
        self.assertEqual(len(listening), 1)
        self.assertIn("listening on right-option · hold to talk, double-tap for hands-free"
                      " · ctrl-opt-d: hold or tap (remote-friendly)", listening[0])

    def test_shutdown_disables_the_tap(self):
        app = run_app(self, lambda app: None)
        self.assertEqual([enable for enable, _ in app.tap_os.enables], [True, False])

    def test_a_shutdown_names_its_reason_and_starts_the_exit_guard(self):
        app = run_app(self, lambda app: None)
        self.assertEqual(app.logged("● shutting down"), ["● shutting down (quit)"])

    def test_without_a_tap_run_exits_with_the_accessibility_hint(self):
        with self.assertRaises(SystemExit) as raised:
            run_app(self, lambda app: self.fail("the main loop ran"), create_ok=False)
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(self.app.logged("failed to create event tap"),
                         ["failed to create event tap — grant Accessibility to this terminal "
                          "(System Settings → Privacy & Security → Accessibility, called "
                          "Device Control and Data Access on newer macOS)"])
        self.assertIsNone(self.app.gate.target, "no poller without a tap")
        self.assertIsNone(self.app.gate.watchdog, "no watchdog without a tap")

    def test_an_unparsable_hotkey_is_logged_and_disabled(self):
        def drive(app):
            app.key_down(D, app.hotkey_flags())
            app.key_up(D)
            self.assertEqual(app.engine.calls, [])
            self.assertEqual(app.controller_kwargs["binding"]["hotkey_spec"], "")

        app = run_app(self, drive, hotkey_spec="ctrl-opt-ü")
        self.assertEqual(app.logged("! unrecognized --hotkey"),
                         ["! unrecognized --hotkey 'ctrl-opt-ü' — hotkey disabled"])
        self.assertNotIn("remote-friendly", app.logged("listening on")[0])


@unittest.skipUnless(sys.platform == "darwin", "drives scrivo.run(), which needs PyObjC; macOS-only")
class TriggerEdgeTests(unittest.TestCase):
    def test_press_and_release_edges_reach_the_engine_once_each(self):
        def drive(app):
            app.press_trigger()
            app.press_trigger()                      # no edge: still down
            app.release_trigger()
            app.release_trigger()                    # no edge: still up
            self.assertEqual(app.engine.calls, ["pressed", "released"])

        run_app(self, drive)

    def test_the_other_sides_twin_is_not_the_trigger(self):
        def drive(app):
            # Right Option's keycode with only Left Option's device bit: the
            # aggregate mask says "option", the per-side bit says "not ours".
            app.press_trigger(device_bit=LEFT_OPTION_BIT)
            self.assertEqual(app.engine.calls, [])

        run_app(self, drive)

    def test_fn_has_no_device_bit_so_the_mask_alone_decides(self):
        def drive(app):
            app.flags_changed(63, Quartz.kCGEventFlagMaskSecondaryFn)
            app.flags_changed(63, 0)
            self.assertEqual(app.engine.calls, ["pressed", "released"])

        run_app(self, drive, trigger="fn")

    def test_another_key_or_modifier_during_a_hold_is_a_chord(self):
        def drive(app):
            app.press_trigger()
            app.key_down(A)                          # ⌥A: a shortcut, not dictation
            app.flags_changed(56, Quartz.kCGEventFlagMaskShift | 0x0002)  # left shift joins
            app.release_trigger()
            self.assertEqual(app.engine.calls, ["pressed", "chorded", "chorded", "released"])

        run_app(self, drive)

    def test_keys_and_modifiers_without_a_hold_are_not_chords(self):
        def drive(app):
            app.key_down(A)
            app.flags_changed(56, Quartz.kCGEventFlagMaskShift | 0x0002)
            self.assertEqual(app.engine.calls, [])

        run_app(self, drive)


@unittest.skipUnless(sys.platform == "darwin", "drives scrivo.run(), which needs PyObjC; macOS-only")
class OwnEventAndTypingTests(unittest.TestCase):
    def test_scrivos_own_tagged_events_are_never_a_chord_typing_or_a_trigger_edge(self):
        def drive(app):
            tag = scrivo.SCRIVO_EVENT_TAG
            app.press_trigger()
            app.key_down(9, Quartz.kCGEventFlagMaskCommand, user_data=tag)   # own ⌘V
            app.key_up(9, Quartz.kCGEventFlagMaskCommand, user_data=tag)
            app.flags_changed(RIGHT_OPTION, 0, user_data=tag)                # own flags change
            app.key_down(D, app.hotkey_flags(), user_data=tag)               # own "hotkey"
            self.assertEqual(app.engine.calls, ["pressed"])
            self.assertEqual(app.keydowns(), 0)

        run_app(self, drive)

    def test_user_keydowns_are_counted_but_not_the_hotkey_press_itself(self):
        def drive(app):
            app.key_down(A)
            app.key_down(S)
            self.assertEqual(app.keydowns(), 2)
            app.key_down(D, app.hotkey_flags())      # starting a dictation is not typing
            self.assertEqual(app.keydowns(), 2)
            app.key_up(D)
            app.key_down(D)                          # a plain "d" is
            app.key_down(D, Quartz.kCGEventFlagMaskControl)  # so is ⌃D
            self.assertEqual(app.keydowns(), 4)
            app.key_up(A)                            # key-ups are not counted
            app.flags_changed(56, Quartz.kCGEventFlagMaskShift | 0x0002)
            self.assertEqual(app.keydowns(), 4)

        run_app(self, drive)

    def test_keys_held_sees_any_modifier_or_the_trigger(self):
        def drive(app):
            keys_held = app.delivery_kwargs["keys_held"]
            self.assertFalse(keys_held())
            for flag in (Quartz.kCGEventFlagMaskCommand, Quartz.kCGEventFlagMaskControl,
                         Quartz.kCGEventFlagMaskAlternate, Quartz.kCGEventFlagMaskShift):
                app.tap_os.flags_state = flag
                self.assertTrue(keys_held(), hex(flag))
            app.tap_os.flags_state = Quartz.kCGEventFlagMaskSecondaryFn
            self.assertFalse(keys_held(), "fn is not a modifier for a right-option binding")

        run_app(self, drive)

    def test_keys_held_counts_an_fn_trigger(self):
        def drive(app):
            app.tap_os.flags_state = Quartz.kCGEventFlagMaskSecondaryFn
            self.assertTrue(app.delivery_kwargs["keys_held"]())

        run_app(self, drive, trigger="fn")


@unittest.skipUnless(sys.platform == "darwin", "drives scrivo.run(), which needs PyObjC; macOS-only")
class HotkeyTests(unittest.TestCase):
    def test_a_tap_starts_and_the_next_tap_finishes(self):
        def drive(app):
            app.key_down(D, app.hotkey_flags())
            app.key_up(D)                            # short: the session keeps running
            self.assertEqual(app.engine.calls, ["force_start"])
            self.assertTrue(app.engine.recording)
            app.key_down(D, app.hotkey_flags())      # the next press ends it
            app.key_up(D)
            self.assertEqual(app.engine.calls, ["force_start", "force_finish"])

        app = run_app(self, drive)
        self.assertEqual(app.logged("● dictation started via hotkey"), ["● dictation started via hotkey"])
        self.assertEqual(app.logged("● finished via hotkey"), ["● finished via hotkey"])

    def test_a_hold_finishes_on_release(self):
        def drive(app):
            app.key_down(D, app.hotkey_flags())
            REAL_SLEEP(scrivo.HOLD_THRESHOLD_S + 0.05)
            app.key_up(D)                            # key-up flags are gone: keycode alone
            self.assertEqual(app.engine.calls, ["force_start", "force_finish"])

        app = run_app(self, drive)
        self.assertEqual(len(app.logged("● finished (held ")), 1)

    def test_a_held_release_after_the_session_already_ended_finishes_nothing(self):
        def drive(app):
            app.key_down(D, app.hotkey_flags())
            app.engine.recording = False             # ended some other way meanwhile
            REAL_SLEEP(scrivo.HOLD_THRESHOLD_S + 0.05)
            app.key_up(D)
            self.assertEqual(app.engine.calls, ["force_start"])

        run_app(self, drive)

    def test_the_release_of_the_finishing_press_is_consumed(self):
        def drive(app):
            app.key_down(D, app.hotkey_flags())
            app.key_up(D)
            app.key_down(D, app.hotkey_flags())      # finishes
            app.engine.recording = True              # a new session started meanwhile
            REAL_SLEEP(scrivo.HOLD_THRESHOLD_S + 0.05)
            app.key_up(D)                            # must not finish that one
            self.assertEqual(app.engine.calls, ["force_start", "force_finish"])

        run_app(self, drive)

    def test_autorepeat_and_wrong_modifiers_do_nothing(self):
        def drive(app):
            app.key_down(D, app.hotkey_flags(), autorepeat=1)
            app.key_down(D, Quartz.kCGEventFlagMaskControl)
            app.key_down(D, app.hotkey_flags() | Quartz.kCGEventFlagMaskShift)
            app.key_down(D, app.hotkey_flags() | Quartz.kCGEventFlagMaskSecondaryFn)  # fn is not compared
            self.assertEqual(app.engine.calls, ["force_start"])

        run_app(self, drive)

    def test_a_stray_key_up_without_an_accepted_key_down_finishes_nothing(self):
        def drive(app):
            app.engine.recording = True              # a trigger-started recording
            REAL_SLEEP(scrivo.HOLD_THRESHOLD_S + 0.05)
            app.key_up(D, app.hotkey_flags())
            app.key_down(D)                          # a plain D press is not the hotkey
            REAL_SLEEP(scrivo.HOLD_THRESHOLD_S + 0.05)
            app.key_up(D)
            self.assertEqual(app.engine.calls, [])
            self.assertTrue(app.engine.recording)

        run_app(self, drive)

    def test_a_refused_start_leaves_its_release_inert(self):
        def drive(app):
            app.engine.start_ok = False
            app.key_down(D, app.hotkey_flags())
            REAL_SLEEP(scrivo.HOLD_THRESHOLD_S + 0.05)
            app.engine.recording = True
            app.key_up(D)
            self.assertEqual(app.engine.calls, ["force_start"])

        app = run_app(self, drive)
        self.assertEqual(app.logged("● dictation started via hotkey"), [])

    def test_the_hotkey_during_a_trigger_hold_is_a_chord_then_a_press(self):
        def drive(app):
            app.press_trigger()
            app.engine.recording = True
            app.key_down(D, app.hotkey_flags())
            self.assertEqual(app.engine.calls, ["pressed", "chorded", "force_finish"])

        run_app(self, drive)

    def test_the_hotkey_key_with_wrong_modifiers_during_a_hold_is_not_a_chord(self):
        def drive(app):
            app.press_trigger()
            app.key_down(D)                          # the hotkey's keycode: never a chord
            self.assertEqual(app.engine.calls, ["pressed"])
            self.assertEqual(app.keydowns(), 1)

        run_app(self, drive)


@unittest.skipUnless(sys.platform == "darwin", "drives scrivo.run(), which needs PyObjC; macOS-only")
class ReviveTests(unittest.TestCase):
    def test_a_disabled_tap_event_re_enables_it_and_reports_health(self):
        for event_type in ("kCGEventTapDisabledByTimeout", "kCGEventTapDisabledByUserInput"):
            with self.subTest(event_type):
                def drive(app, event_type=event_type):
                    app.tap_os.enabled = False
                    app.send(getattr(Quartz, event_type), event(0))
                    self.assertEqual([enable for enable, _ in app.tap_os.enables], [True, True])
                    self.assertTrue(app.tap_os.enabled)

                app = run_app(self, drive)
                self.assertEqual(app.logged("! event tap re-enabled after"),
                                 ["! event tap re-enabled after system disable"])
                self.assertEqual(app.tap_health(), [True])

    def test_a_release_missed_while_deaf_is_synthesized(self):
        def drive(app):
            app.press_trigger()
            app.tap_os.flags_state = 0               # the key went up while the tap was off
            app.send(Quartz.kCGEventTapDisabledByTimeout, event(0))
            self.assertEqual(app.engine.calls, ["pressed", "released"])
            app.release_trigger()                    # the late edge is no second release
            self.assertEqual(app.engine.calls, ["pressed", "released"])

        run_app(self, drive)

    def test_a_press_missed_while_deaf_is_adopted_without_starting(self):
        def drive(app):
            app.tap_os.flags_state = Quartz.kCGEventFlagMaskAlternate | RIGHT_OPTION_BIT
            app.send(Quartz.kCGEventTapDisabledByTimeout, event(0))
            app.press_trigger()                      # already down: no edge
            app.release_trigger()
            self.assertEqual(app.engine.calls, ["released"])

        run_app(self, drive)

    def test_a_tap_that_stays_disabled_warns_once_a_minute(self):
        def drive(app):
            app.tap_os.stay_disabled = True
            app.send(Quartz.kCGEventTapDisabledByUserInput, event(0))
            app.send(Quartz.kCGEventTapDisabledByUserInput, event(0))

        app = run_app(self, drive)
        self.assertEqual(app.logged("✗ event tap stays disabled"),
                         ["✗ event tap stays disabled — re-grant Accessibility to Scrivo"])
        self.assertEqual(app.logged("! event tap re-enabled"), [])
        self.assertEqual(app.tap_health(), [False])

    def test_waking_from_sleep_resets_the_gesture_and_re_enables_the_tap(self):
        def drive(app):
            app.status.on_wake()
            self.assertEqual(app.engine.calls, ["force_reset"])
            self.assertEqual([enable for enable, _ in app.tap_os.enables], [True, True])

        run_app(self, drive)


@unittest.skipUnless(sys.platform == "darwin", "drives scrivo.run(), which needs PyObjC; macOS-only")
class PollerTests(unittest.TestCase):
    def test_each_poll_sleeps_a_second_and_ticks_the_capture(self):
        def drive(app):
            app.gate.tick()
            app.gate.tick()
            self.assertEqual(app.gate.sleeps, [1.0, 1.0, 1.0])
            self.assertEqual(app.captures[0].ticks, 2)
            self.assertEqual(app.engine.calls, [])

        run_app(self, drive)

    def test_a_silently_disabled_tap_is_revived_by_the_watchdog_never_via_callafter(self):
        def drive(app):
            app.tap_os.enabled = False               # no notification, no callback
            app.gate.tick()                          # the capture poller no longer does it
            self.assertEqual([enable for enable, _ in app.tap_os.enables], [True])
            app.gate.watchdog.__self__.watchdog_pass()
            enable, thread = app.tap_os.enables[-1]
            self.assertTrue(enable)
            self.assertIs(thread, threading.current_thread(), "revive_tap runs where the watchdog pass runs")
            self.assertIsNot(thread, app.gate.thread, "...not on the capture poller")
            self.assertNotIn("revive_tap", app.called_after)

        app = run_app(self, drive)
        self.assertEqual(app.logged("! event tap re-enabled after"),
                         ["! event tap re-enabled after silent disable"])

    def test_a_lost_release_is_synthesized_after_two_physically_up_polls(self):
        def drive(app):
            app.press_trigger()
            app.engine.recording = True
            app.gate.tick()                          # one miss
            self.assertEqual(app.engine.calls, ["pressed"])
            app.gate.tick()                          # two: release
            self.assertEqual(app.engine.calls, ["pressed", "released"])
            app.release_trigger()                    # trigger state was cleared: no second edge
            self.assertEqual(app.engine.calls, ["pressed", "released"])

        app = run_app(self, drive)
        self.assertEqual(app.logged("! lost release recovered"),
                         ["! lost release recovered — finishing recording"])

    def test_a_physically_down_poll_resets_the_count(self):
        def drive(app):
            app.engine.recording = True
            app.gate.tick()
            app.tap_os.flags_state = Quartz.kCGEventFlagMaskAlternate | RIGHT_OPTION_BIT
            app.gate.tick()
            app.tap_os.flags_state = 0
            app.gate.tick()
            self.assertEqual(app.engine.calls, [])
            app.gate.tick()
            self.assertEqual(app.engine.calls, ["released"])

        run_app(self, drive)

    def test_hands_free_is_exempt(self):
        def drive(app):
            app.engine.recording = app.engine.hands_free = True
            for _ in range(4):
                app.gate.tick()
            self.assertEqual(app.engine.calls, [])

        run_app(self, drive)

    def test_not_recording_is_never_a_lost_release(self):
        def drive(app):
            for _ in range(3):
                app.gate.tick()
            self.assertEqual(app.engine.calls, [])

        run_app(self, drive)

    def test_the_poller_stops_once_shutdown_is_requested(self):
        def drive(app):
            app.boundaries[0].request()
            app.gate.stop()
            self.assertFalse(app.gate.thread.is_alive())
            self.assertEqual(app.captures[0].ticks, 0)

        run_app(self, drive)


@unittest.skipUnless(sys.platform == "darwin", "drives scrivo.run(), which needs PyObjC; macOS-only")
class RebindTests(unittest.TestCase):
    def test_a_new_trigger_takes_over_and_resets_the_gesture(self):
        def drive(app):
            app.press_trigger()
            app.controller_kwargs["rebind"](new_trigger="right-cmd")
            self.assertEqual(app.engine.calls, ["pressed", "force_reset"])
            self.assertEqual(app.controller_kwargs["binding"]["trigger"], "right-cmd")
            app.release_trigger()                    # the old key is no trigger now
            app.key_down(A)                          # and no hold is in progress
            self.assertEqual(app.engine.calls, ["pressed", "force_reset"])
            app.flags_changed(54, Quartz.kCGEventFlagMaskCommand | 0x0010)
            app.flags_changed(54, 0)
            self.assertEqual(app.engine.calls, ["pressed", "force_reset", "pressed", "released"])
            app.tap_os.flags_state = Quartz.kCGEventFlagMaskCommand | 0x0010
            app.engine.recording = True
            app.gate.tick()
            app.gate.tick()                          # physically down on the new trigger: no release
            self.assertEqual(app.engine.calls[-1], "released")
            self.assertEqual(len(app.engine.calls), 4)

        app = run_app(self, drive)
        self.assertEqual(app.logged("● trigger is now"), ["● trigger is now right-cmd"])

    def test_the_hands_free_hint_follows_the_new_trigger(self):
        def drive(app):
            app.controller_kwargs["rebind"](new_trigger="left-cmd")
            self.assertEqual(scrivo.hands_free_hint(app.controller_kwargs["binding"]["trigger"]),
                             "tap left ⌘ to stop")

        run_app(self, drive)

    def test_a_new_hotkey_replaces_the_old_one(self):
        def drive(app):
            app.key_down(D, app.hotkey_flags())      # pressed under the old hotkey...
            app.controller_kwargs["rebind"](new_hotkey_spec="f13")
            self.assertEqual(app.controller_kwargs["binding"]["hotkey_spec"], "f13")
            REAL_SLEEP(scrivo.HOLD_THRESHOLD_S + 0.05)
            app.key_up(F13)                          # ...its press state is dropped
            self.assertEqual(app.engine.calls, ["force_start"])
            app.key_down(D, app.hotkey_flags())      # the old chord is plain typing now
            self.assertEqual(app.keydowns(), 1)
            app.key_down(F13)
            self.assertEqual(app.engine.calls, ["force_start", "force_finish"])

        app = run_app(self, drive)
        self.assertEqual(app.logged("● alternate hotkey"), ["● alternate hotkey is now f13"])

    def test_an_empty_or_unparsable_hotkey_turns_it_off(self):
        for spec in ("", "ctrl-opt-ü"):
            with self.subTest(spec=spec):
                def drive(app, spec=spec):
                    app.controller_kwargs["rebind"](new_hotkey_spec=spec)
                    self.assertEqual(app.controller_kwargs["binding"]["hotkey_spec"], "")
                    app.key_down(D, app.hotkey_flags())
                    app.key_up(D)
                    self.assertEqual(app.engine.calls, [])
                    self.assertEqual(app.keydowns(), 1)

                app = run_app(self, drive)
                self.assertEqual(app.logged("● alternate hotkey"), ["● alternate hotkey off"])


@unittest.skipUnless(sys.platform == "darwin", "drives scrivo.run(), which needs PyObjC; macOS-only")
class ShutdownTests(unittest.TestCase):
    def test_once_shutdown_is_requested_every_event_passes_through_untouched(self):
        def drive(app):
            app.press_trigger()
            app.boundaries[0].request()
            app.engine.recording = True
            app.release_trigger()
            app.key_down(A)
            app.key_down(D, app.hotkey_flags())
            app.key_up(D)
            app.tap_os.enabled = False
            app.send(Quartz.kCGEventTapDisabledByTimeout, event(0))
            self.assertEqual(app.engine.calls, ["pressed"])
            self.assertEqual(app.keydowns(), 0)

        app = run_app(self, drive)
        self.assertEqual(app.logged("! event tap re-enabled"), [])


if __name__ == "__main__":
    unittest.main()
