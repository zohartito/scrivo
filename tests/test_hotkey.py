"""HotkeyListener with a fake Quartz adapter: the event tap's decisions without
macOS (these run on Windows too). The macOS-only tests pin hotkey.py's literal
flag masks, and this fake's constants, to Quartz's own values.

tests/test_hotkey_tap.py drives the same decisions through the real run().
"""
import ast
import signal
import subprocess
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import hotkey
from gestures import HOLD_THRESHOLD_S

ROOT = Path(__file__).resolve().parents[1]
CMD, CTRL, OPT, SHIFT, FN = (hotkey.kCGEventFlagMaskCommand, hotkey.kCGEventFlagMaskControl,
                             hotkey.kCGEventFlagMaskAlternate, hotkey.kCGEventFlagMaskShift,
                             hotkey.kCGEventFlagMaskSecondaryFn)
RIGHT_OPTION, RIGHT_OPTION_BIT, LEFT_OPTION_BIT = 61, 0x0040, 0x0020
D, A = 2, 0
TAP = object()


class FakeQuartz:
    """The Quartz names HotkeyListener uses. Integer constants are Quartz's
    (QuartzConstantTests checks them on macOS); events are dicts."""
    kCGEventKeyDown = 10
    kCGEventKeyUp = 11
    kCGEventFlagsChanged = 12
    kCGEventTapDisabledByTimeout = 0xFFFFFFFE
    kCGEventTapDisabledByUserInput = 0xFFFFFFFF
    kCGKeyboardEventAutorepeat = 8
    kCGKeyboardEventKeycode = 9
    kCGEventSourceStateCombinedSessionState = 0
    kCGSessionEventTap = 1
    kCGHeadInsertEventTap = 0
    kCGEventTapOptionListenOnly = 1
    kCFRunLoopCommonModes = "kCFRunLoopCommonModes"
    INTEGER_CONSTANTS = ("kCGEventKeyDown", "kCGEventKeyUp", "kCGEventFlagsChanged",
                         "kCGEventTapDisabledByTimeout", "kCGEventTapDisabledByUserInput",
                         "kCGKeyboardEventAutorepeat", "kCGKeyboardEventKeycode",
                         "kCGEventSourceStateCombinedSessionState", "kCGSessionEventTap",
                         "kCGHeadInsertEventTap", "kCGEventTapOptionListenOnly")

    def __init__(self, *, create_ok=True):
        self.create_ok = create_ok
        self.flags_state = 0
        self.enabled = False
        self.stay_disabled = False
        self.enables = []
        self.created = []
        self.sources = []

    def CGEventGetIntegerValueField(self, event, field):
        return {self.kCGKeyboardEventKeycode: event["keycode"],
                self.kCGKeyboardEventAutorepeat: event["autorepeat"]}[field]

    def CGEventGetFlags(self, event):
        return event["flags"]

    def CGEventSourceFlagsState(self, state):
        assert state == self.kCGEventSourceStateCombinedSessionState
        return self.flags_state

    def CGEventTapEnable(self, tap, enable):
        self.enables.append(bool(enable))
        self.enabled = bool(enable) and not self.stay_disabled

    def CGEventTapIsEnabled(self, tap):
        return self.enabled

    def CGEventMaskBit(self, event_type):
        return 1 << event_type

    def CGEventTapCreate(self, tap, place, options, mask, callback, refcon):
        self.created.append((tap, place, options, mask, callback, refcon))
        return TAP if self.create_ok else None

    def CFMachPortCreateRunLoopSource(self, allocator, tap, order):
        return ("source", tap, order)

    def CFRunLoopGetCurrent(self):
        return "current-loop"

    def CFRunLoopAddSource(self, loop, source, mode):
        self.sources.append((loop, source, mode))


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.recording = self.hands_free = False
        self.start_ok = True

    def snapshot(self):
        return self.recording, self.hands_free

    def __getattr__(self, name):
        if name in ("pressed", "released", "chorded", "force_reset"):
            return lambda: self.calls.append(name)
        raise AttributeError(name)

    def force_start(self):
        self.calls.append("force_start")
        if self.start_ok and not self.recording:
            self.recording = True
            return True
        return False

    def force_finish(self):
        self.calls.append("force_finish")
        was, self.recording = self.recording, False
        return was


class World:
    """One listener, its fakes and a hand-advanced clock."""

    def __init__(self, trigger="right-option", hotkey_spec="ctrl-opt-d", *, create_ok=True):
        self.quartz = FakeQuartz(create_ok=create_ok)
        self.engine = FakeEngine()
        self.capture = types.SimpleNamespace(ticks=0)
        self.capture.tick = lambda: setattr(self.capture, "ticks", self.capture.ticks + 1)
        self.stopping = False
        self.shutdown = types.SimpleNamespace(requested=lambda: self.stopping)
        self.logs, self.health, self.sleeps = [], [], []
        self.now = 1000.0
        self.keydowns = {"count": 0}
        self.status_ui = types.SimpleNamespace(set_tap_health=self.health.append)
        self.listener = hotkey.HotkeyListener(
            trigger, hotkey_spec, quartz=self.quartz, engine=self.engine, capture=self.capture,
            shutdown=self.shutdown, is_own_event=lambda event: event["own"], user_keydowns=self.keydowns,
            log=self.logs.append, status_ui=self.status_ui, ui_call=lambda method, *args: method(*args),
            clock=lambda: self.now, sleep=self.sleep)
        self.listener.tap = TAP                      # as if start() had run
        self.quartz.enabled = True
        self.polls_left = 0

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        if self.polls_left == 0:
            self.stopping = True
        self.polls_left -= 1

    def poll(self, times):
        """Run the poller for exactly ``times`` polls, then it sees shutdown."""
        self.polls_left, self.stopping = times, False
        self.listener.resync_poller()
        self.stopping = False

    def watch(self, times):
        """The same for the tap watchdog thread's loop."""
        self.polls_left, self.stopping = times, False
        self.listener.tap_watchdog()
        self.stopping = False

    def send(self, event_type, keycode=0, flags=0, *, autorepeat=0, own=False):
        event = {"keycode": keycode, "flags": flags, "autorepeat": autorepeat, "own": own}
        returned = self.listener.callback(None, event_type, event, None)
        assert returned is event, "the tap is listen-only: it hands every event back"
        return event

    def down(self, keycode, flags=0, **kw):
        return self.send(FakeQuartz.kCGEventKeyDown, keycode, flags, **kw)

    def up(self, keycode, flags=0, **kw):
        return self.send(FakeQuartz.kCGEventKeyUp, keycode, flags, **kw)

    def flags(self, keycode, flags, **kw):
        return self.send(FakeQuartz.kCGEventFlagsChanged, keycode, flags, **kw)

    def press(self, bit=RIGHT_OPTION_BIT):
        self.flags(RIGHT_OPTION, OPT | bit)

    def release(self):
        self.flags(RIGHT_OPTION, 0)


@unittest.skipUnless(sys.platform == "darwin", "Quartz exists only on macOS")
class QuartzConstantTests(unittest.TestCase):
    def test_the_literal_flag_masks_are_quartzs(self):
        import Quartz
        for name in ("kCGEventFlagMaskShift", "kCGEventFlagMaskControl", "kCGEventFlagMaskAlternate",
                     "kCGEventFlagMaskCommand", "kCGEventFlagMaskSecondaryFn"):
            with self.subTest(name):
                self.assertEqual(getattr(hotkey, name), getattr(Quartz, name))

    def test_the_tables_equal_the_ones_built_from_quartz(self):
        import Quartz
        q = Quartz
        # As scrivo.py built them before the move (9efe54b).
        self.assertEqual(hotkey.TRIGGERS, {
            "fn": (q.kCGEventFlagMaskSecondaryFn, 63, 0),
            "right-cmd": (q.kCGEventFlagMaskCommand, 54, 0x0010),
            "right-option": (q.kCGEventFlagMaskAlternate, 61, 0x0040),
            "right-shift": (q.kCGEventFlagMaskShift, 60, 0x0004),
            "right-ctrl": (q.kCGEventFlagMaskControl, 62, 0x2000),
            "left-cmd": (q.kCGEventFlagMaskCommand, 55, 0x0008),
            "left-option": (q.kCGEventFlagMaskAlternate, 58, 0x0020),
            "left-shift": (q.kCGEventFlagMaskShift, 56, 0x0002),
            "left-ctrl": (q.kCGEventFlagMaskControl, 59, 0x0001),
        })
        self.assertEqual(hotkey.MODIFIER_FLAGS, {
            "cmd": q.kCGEventFlagMaskCommand, "ctrl": q.kCGEventFlagMaskControl,
            "opt": q.kCGEventFlagMaskAlternate, "alt": q.kCGEventFlagMaskAlternate,
            "shift": q.kCGEventFlagMaskShift})
        self.assertEqual(hotkey.ALL_MODIFIERS, q.kCGEventFlagMaskCommand | q.kCGEventFlagMaskControl
                         | q.kCGEventFlagMaskAlternate | q.kCGEventFlagMaskShift)

    def test_this_fake_uses_quartzs_constants(self):
        import Quartz
        for name in FakeQuartz.INTEGER_CONSTANTS:
            with self.subTest(name):
                self.assertEqual(getattr(FakeQuartz, name), getattr(Quartz, name))
        for event_type in (Quartz.kCGEventKeyDown, Quartz.kCGEventFlagsChanged):
            self.assertEqual(FakeQuartz().CGEventMaskBit(event_type), Quartz.CGEventMaskBit(event_type))


class ImportTests(unittest.TestCase):
    def test_hotkey_imports_without_pyobjc_or_scrivo(self):
        tree = ast.parse((ROOT / "hotkey.py").read_text(encoding="utf-8"))
        imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                    for alias in node.names}
        imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                     if isinstance(node, ast.ImportFrom) and node.module}
        self.assertFalse(imported & {"Quartz", "AppKit", "objc", "PyObjCTools", "scrivo"}, imported)
        blocked = "import sys; [sys.modules.__setitem__(m, None) for m in ('Quartz', 'AppKit', 'objc')]; import hotkey"
        result = subprocess.run([sys.executable, "-c", blocked], cwd=ROOT, capture_output=True, text=True,
                                timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)


class TriggerTests(unittest.TestCase):
    def test_edges_need_the_per_side_device_bit_and_fire_once(self):
        world = World()
        world.press(LEFT_OPTION_BIT)                 # the other side's twin
        self.assertEqual(world.engine.calls, [])
        world.press()
        world.press()
        world.release()
        world.release()
        self.assertEqual(world.engine.calls, ["pressed", "released"])

    def test_keys_and_modifiers_during_a_hold_are_chords(self):
        world = World()
        world.down(A)
        world.flags(56, SHIFT | 0x0002)
        self.assertEqual(world.engine.calls, [])
        world.press()
        world.down(A)
        world.flags(56, SHIFT | 0x0002)
        self.assertEqual(world.engine.calls, ["pressed", "chorded", "chorded"])

    def test_own_events_are_never_chords_typing_or_edges(self):
        world = World()
        world.press()
        world.down(9, CMD, own=True)
        world.up(9, CMD, own=True)
        world.flags(RIGHT_OPTION, 0, own=True)
        world.down(D, CTRL | OPT, own=True)
        self.assertEqual(world.engine.calls, ["pressed"])
        self.assertEqual(world.keydowns["count"], 0)

    def test_user_keydowns_count_everything_but_the_hotkey_press(self):
        world = World()
        world.down(A)
        world.down(D, CTRL | OPT)
        world.down(D, CTRL)
        world.down(D)
        world.up(A)
        self.assertEqual(world.keydowns["count"], 3)

    def test_keys_held_sees_any_modifier_or_the_trigger(self):
        world = World()
        self.assertFalse(world.listener.keys_held())
        for flag in (CMD, CTRL, OPT, SHIFT):
            world.quartz.flags_state = flag
            self.assertTrue(world.listener.keys_held())
        world.quartz.flags_state = FN
        self.assertFalse(world.listener.keys_held())
        fn_world = World("fn")
        fn_world.quartz.flags_state = FN
        self.assertTrue(fn_world.listener.keys_held())


class HotkeyTests(unittest.TestCase):
    def test_tap_then_tap(self):
        world = World()
        world.down(D, CTRL | OPT)
        world.now += HOLD_THRESHOLD_S - 0.01
        world.up(D)
        self.assertEqual(world.engine.calls, ["force_start"])
        world.down(D, CTRL | OPT)
        world.engine.recording = True                # a new session started meanwhile
        world.now += 5
        world.up(D)                                  # the finishing press's release is consumed
        self.assertEqual(world.engine.calls, ["force_start", "force_finish"])
        self.assertEqual(world.logs, ["● dictation started via hotkey", "● finished via hotkey"])

    def test_a_hold_of_exactly_the_threshold_finishes(self):
        world = World()
        world.now = 0.0                              # so now - pressed_at is exactly the threshold
        world.down(D, CTRL | OPT)
        world.now += HOLD_THRESHOLD_S
        world.up(D, 0)                               # key-up flags are gone: keycode alone
        self.assertEqual(world.engine.calls, ["force_start", "force_finish"])
        self.assertRegex(world.logs[-1], r"^● finished \(held 0\.\ds\)$")

    def test_autorepeat_wrong_modifiers_and_stray_key_ups_do_nothing(self):
        world = World()
        world.engine.recording = True
        world.down(D, CTRL | OPT, autorepeat=1)
        world.down(D, CTRL)
        world.down(D, CTRL | OPT | SHIFT)
        world.now += 10
        world.up(D, CTRL | OPT)
        self.assertEqual(world.engine.calls, [])

    def test_a_refused_start_records_no_press(self):
        world = World()
        world.engine.start_ok = False
        world.down(D, CTRL | OPT)
        world.engine.recording = True
        world.now += 10
        world.up(D)
        self.assertEqual(world.engine.calls, ["force_start"])

    def test_the_hotkey_during_a_trigger_hold_is_a_chord(self):
        world = World()
        world.press()
        world.down(D, CTRL | OPT)
        self.assertEqual(world.engine.calls, ["pressed", "chorded", "force_start"])

    def test_an_unparsable_hotkey_is_off_and_logged(self):
        world = World(hotkey_spec="ctrl-opt-ü")
        self.assertIsNone(world.listener.hotkey)
        self.assertEqual(world.listener.binding["hotkey_spec"], "")
        self.assertEqual(world.logs, ["! unrecognized --hotkey 'ctrl-opt-ü' — hotkey disabled"])
        world.down(D, CTRL | OPT)
        self.assertEqual(world.engine.calls, [])
        self.assertEqual(world.keydowns["count"], 1)

    def test_no_hotkey_spec_is_off_silently(self):
        world = World(hotkey_spec="")
        self.assertIsNone(world.listener.hotkey)
        self.assertEqual(world.logs, [])


class ReviveTests(unittest.TestCase):
    def test_a_disabled_tap_event_re_enables_resyncs_and_reports(self):
        for event_type in (FakeQuartz.kCGEventTapDisabledByTimeout, FakeQuartz.kCGEventTapDisabledByUserInput):
            with self.subTest(event_type=event_type):
                world = World()
                world.press()
                world.send(event_type)               # the key went up while deaf (flags_state 0)
                self.assertEqual(world.quartz.enables, [True])
                self.assertEqual(world.engine.calls, ["pressed", "released"])
                self.assertEqual(world.logs, ["! event tap re-enabled after system disable"])
                self.assertEqual(world.health, [True])

    def test_a_stuck_tap_warns_again_only_after_sixty_seconds(self):
        world = World()
        world.quartz.stay_disabled = True
        world.send(FakeQuartz.kCGEventTapDisabledByUserInput)
        world.now += 60
        world.send(FakeQuartz.kCGEventTapDisabledByUserInput)
        world.now += 0.5
        world.send(FakeQuartz.kCGEventTapDisabledByUserInput)
        self.assertEqual(world.logs, ["✗ event tap stays disabled — re-grant Accessibility to Scrivo"] * 2)
        self.assertEqual(world.health, [False, False])

    def test_without_a_menu_bar_health_is_only_logged(self):
        world = World()
        world.listener.status_ui = None
        world.send(FakeQuartz.kCGEventTapDisabledByTimeout)
        self.assertEqual(world.health, [])
        self.assertEqual(world.logs, ["! event tap re-enabled after system disable"])


class PollerTests(unittest.TestCase):
    def test_two_physically_up_polls_synthesize_the_release(self):
        world = World()
        world.press()
        world.engine.recording = True
        world.poll(1)
        self.assertEqual(world.engine.calls, ["pressed"])
        world.poll(2)                                # the miss count restarts with the loop
        self.assertEqual(world.engine.calls, ["pressed", "released"])
        self.assertFalse(world.listener.trigger_down)
        self.assertEqual(world.logs, ["! lost release recovered — finishing recording"])
        self.assertEqual(world.capture.ticks, 3)
        self.assertEqual(set(world.sleeps), {1.0})

    def test_hands_free_and_idle_are_exempt(self):
        world = World()
        world.poll(3)
        world.engine.recording = world.engine.hands_free = True
        world.poll(3)
        self.assertEqual(world.engine.calls, [])

    def test_a_physically_down_poll_resets_the_count(self):
        world = World()
        world.engine.recording = True
        world.quartz.flags_state = OPT | RIGHT_OPTION_BIT
        world.poll(3)
        self.assertEqual(world.engine.calls, [])

    def test_a_failed_round_is_logged_and_the_loop_keeps_polling(self):
        world = World()
        rounds = iter([RuntimeError("boom"), None])

        def snapshot():
            failure = next(rounds)
            if failure:
                raise failure
            return False, False

        world.engine.snapshot = snapshot
        world.poll(2)
        self.assertEqual(world.capture.ticks, 2)     # the second round ran after the first failed
        self.assertEqual(world.logs, ["! resync poller round failed: RuntimeError: boom"])

    def test_every_round_stamps_the_beat_the_watchdog_reads(self):
        world = World()
        self.assertIsNone(world.listener.poller_beat)
        world.now = 1234.0
        world.poll(1)
        self.assertEqual(world.listener.poller_beat, 1234.0)

    def test_shutdown_stops_it_before_the_next_poll(self):
        world = World()
        world.stopping = True
        world.listener.resync_poller()
        self.assertEqual((world.sleeps, world.capture.ticks), ([], 0))
        world.poll(0)
        self.assertEqual((world.sleeps, world.capture.ticks), ([1.0], 0))


class WatchdogTests(unittest.TestCase):
    """The tap comes back on a thread of its own (2026-10-19: tap disabled, hotkey
    dead for good, process idle and healthy, zero revive log lines: the revive
    lived on the poller behind capture.tick(), which can block on CoreAudio)."""

    def test_a_silently_disabled_tap_is_revived_by_the_watchdog_itself(self):
        world = World()
        world.quartz.enabled = False
        with mock.patch("threading.Thread") as thread:
            world.watch(1)
        thread.assert_not_called()                   # on its own thread: never via callAfter
        self.assertEqual(world.quartz.enables, [True])
        self.assertEqual(world.logs, ["! event tap re-enabled after silent disable"])
        self.assertEqual(world.capture.ticks, 0)     # it never touches the capture service

    def test_a_healthy_tap_is_left_alone(self):
        world = World()
        world.watch(3)
        self.assertEqual((world.quartz.enables, world.logs, set(world.sleeps)), ([], [], {1.0}))

    def test_the_poller_no_longer_revives_the_tap(self):
        world = World()
        world.quartz.enabled = False
        world.poll(2)
        self.assertEqual(world.quartz.enables, [])

    def test_the_tap_comes_back_while_a_capture_tick_never_returns(self):
        world = World()
        entered, release = threading.Event(), threading.Event()
        world.capture.tick = lambda: (entered.set(), release.wait(20))
        world.listener.sleep = lambda seconds: time.sleep(0.01)
        threads_before = threading.active_count()
        world.listener.start()                       # the real poller and watchdog threads
        try:
            self.assertTrue(entered.wait(5), "the poller never reached the blocking tick")
            world.quartz.enabled = False             # macOS silently disables the tap
            self.wait_for(lambda: "! event tap re-enabled after silent disable" in world.logs)
            self.assertEqual(world.quartz.enables, [True, True], "start() enables it, then the watchdog does")
        finally:
            world.stopping = True
            release.set()
            self.wait_for(lambda: threading.active_count() <= threads_before)  # both threads end

    @staticmethod
    def wait_for(condition, seconds=5.0):
        deadline = time.monotonic() + seconds
        while not condition() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert condition(), "timed out waiting"

    def test_a_blocked_release_resync_cannot_stall_the_watchdog(self):
        # finishing a capture can wait on the capture lock; the revive must not
        world = World()
        world.press()                                # trigger_down, then the key goes up while deaf
        world.quartz.flags_state = 0
        world.quartz.enabled = False
        stuck, release = threading.Event(), threading.Event()
        world.engine.released = lambda: (stuck.set(), release.wait(20))
        try:
            finished = threading.Event()
            threading.Thread(target=lambda: (world.watch(1), finished.set()), daemon=True).start()
            self.assertTrue(finished.wait(5), "the watchdog was held up by the blocked release resync")
            self.assertEqual(world.quartz.enables, [True])
            self.assertEqual(world.logs, ["! event tap re-enabled after silent disable"])
            self.assertTrue(stuck.wait(5), "the release edge was never delivered")
            world.quartz.enabled = False             # and the next disable is still repaired
            world.watch(1)
            self.assertEqual(world.quartz.enables, [True, True])
        finally:
            release.set()

    def test_a_press_that_arrived_during_the_resync_keeps_its_gesture(self):
        world = World()
        world.press()
        world.quartz.flags_state = OPT | RIGHT_OPTION_BIT   # pressed again by the time the helper runs
        world.listener.release_missed_edge(False)
        self.assertEqual(world.engine.calls, ["pressed"])
        world.quartz.flags_state = 0
        world.listener.release_missed_edge(False)
        self.assertEqual(world.engine.calls, ["pressed", "released"])

    def test_the_poller_rechecks_under_the_lock_before_releasing(self):
        world = World()
        world.press()
        world.engine.recording = True
        calls = []
        real_snapshot = world.engine.snapshot

        def snapshot():
            calls.append(1)
            if len(calls) == 3:                      # between the second miss and the release:
                world.engine.recording = False       # the revive already finished the old recording
            return real_snapshot()

        world.engine.snapshot = snapshot
        world.poll(2)
        self.assertEqual(world.engine.calls, ["pressed"])
        self.assertEqual(world.logs, [])

    def test_a_repeating_round_failure_is_logged_once_a_minute(self):
        world = World()

        def broken():
            raise RuntimeError("boom")

        world.engine.snapshot = broken
        world.poll(3)
        self.assertEqual(world.logs, ["! resync poller round failed: RuntimeError: boom"])
        world.now += 61
        world.poll(1)
        self.assertEqual(len(world.logs), 2)

    def test_a_pass_after_shutdown_leaves_a_deliberately_disabled_tap_alone(self):
        world = World()
        world.quartz.enabled = False
        world.stopping = True
        world.listener.watchdog_pass()
        self.assertEqual(world.quartz.enables, [])

    def test_a_failing_revive_is_logged_once_a_minute_and_the_watchdog_lives(self):
        world = World()
        world.quartz.enabled = False

        def refuse(tap, enable):
            raise RuntimeError("boom")

        world.quartz.CGEventTapEnable = refuse
        world.watch(2)
        self.assertEqual(world.logs, ["! hotkey watchdog failed: RuntimeError: boom"])
        world.now += 61
        world.watch(1)
        self.assertEqual(world.logs, ["! hotkey watchdog failed: RuntimeError: boom"] * 2)

    def test_a_stalled_poller_is_reported_once_a_minute_with_the_pid(self):
        world = World()
        world.listener.poller_beat = world.now - 31
        world.watch(1)
        self.assertEqual(len(world.logs), 1)
        self.assertRegex(world.logs[0], r"^! capture poller stalled for 31s .*kill -USR1 \d+ writes every thread")
        world.watch(2)                               # the same minute: silent
        self.assertEqual(len(world.logs), 1)
        world.now += 61
        world.listener.poller_beat = world.now - 90
        world.watch(1)
        self.assertEqual(len(world.logs), 2)
        self.assertIn("stalled for 90s", world.logs[1])

    def test_a_poller_that_is_merely_between_rounds_or_not_started_is_not_stalled(self):
        world = World()
        world.watch(1)                               # never started: no beat
        world.listener.poller_beat = world.now - 29
        world.watch(1)
        self.assertEqual(world.logs, [])

    def test_shutdown_stops_it_before_the_next_check(self):
        world = World()
        world.quartz.enabled = False
        world.stopping = True
        world.listener.tap_watchdog()
        self.assertEqual((world.sleeps, world.quartz.enables), ([], []))
        world.watch(0)
        self.assertEqual((world.sleeps, world.quartz.enables), ([1.0], []))


@unittest.skipUnless(hasattr(signal, "SIGUSR1"), "POSIX signals")
class StackDumpTests(unittest.TestCase):
    def test_sigusr1_writes_every_threads_stack_to_stderr(self):
        code = (
            "import os, signal, sys, threading, time\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "import scrivo_log\n"
            "scrivo_log.enable_stack_dump_signal()\n"
            "def stuck_in_capture_tick():\n"
            "    threading.Event().wait()\n"
            "threading.Thread(target=stuck_in_capture_tick, daemon=True).start()\n"
            "time.sleep(0.3)\n"
            "os.kill(os.getpid(), signal.SIGUSR1)\n"
            "time.sleep(0.5)\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("stuck_in_capture_tick", result.stderr)
        self.assertIn("(most recent call first)", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_a_stderr_without_a_file_descriptor_never_stops_startup(self):
        import scrivo_log

        class NoDescriptor:
            def write(self, text):
                return len(text)

            def flush(self):
                pass

        for bad_stderr in (NoDescriptor(), None):
            with self.subTest(stderr=bad_stderr), mock.patch.object(sys, "stderr", bad_stderr):
                scrivo_log.enable_stack_dump_signal()   # must not raise

    def test_registering_twice_is_harmless(self):
        import scrivo_log
        scrivo_log.enable_stack_dump_signal()
        scrivo_log.enable_stack_dump_signal()


class StartTests(unittest.TestCase):
    def test_start_creates_a_listen_only_tap_adds_it_enables_it_and_starts_the_poller(self):
        world = World()
        world.listener.tap = None
        with mock.patch("threading.Thread") as thread:
            world.listener.start()
        q = FakeQuartz
        mask = (1 << q.kCGEventFlagsChanged) | (1 << q.kCGEventKeyDown) | (1 << q.kCGEventKeyUp)
        self.assertEqual(world.quartz.created, [(q.kCGSessionEventTap, q.kCGHeadInsertEventTap,
                                                 q.kCGEventTapOptionListenOnly, mask,
                                                 world.listener.callback, None)])
        self.assertIs(world.listener.tap, TAP)
        self.assertEqual(world.quartz.sources, [("current-loop", ("source", TAP, 0), q.kCFRunLoopCommonModes)])
        self.assertEqual(world.quartz.enables, [True])
        self.assertEqual(thread.call_args_list, [
            mock.call(target=world.listener.resync_poller, daemon=True),
            mock.call(target=world.listener.tap_watchdog, daemon=True, name="scrivo-tap-watchdog")])
        self.assertEqual(thread.return_value.start.call_count, 2)
        self.assertEqual(world.listener.poller_beat, world.now)  # so a poller that never runs is noticed

    def test_a_refused_tap_exits_with_the_accessibility_hint(self):
        world = World(create_ok=False)
        with mock.patch("threading.Thread") as thread, self.assertRaises(SystemExit) as raised:
            world.listener.start()
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("failed to create event tap", world.logs[-1])
        thread.assert_not_called()


class RebindAndShutdownTests(unittest.TestCase):
    def test_rebind_trigger(self):
        world = World()
        world.press()
        world.listener.rebind(new_trigger="right-cmd")
        self.assertEqual(world.listener.binding,
                         {"trigger": "right-cmd", "hotkey_spec": "ctrl-opt-d",
                          "mask": CMD, "keycode": 54, "device_bit": 0x0010})
        self.assertFalse(world.listener.trigger_down)
        world.release()                              # the old key no longer counts
        world.flags(54, CMD | 0x0010)
        self.assertEqual(world.engine.calls, ["pressed", "force_reset", "pressed"])
        self.assertEqual(world.logs, ["● trigger is now right-cmd"])

    def test_rebind_hotkey_drops_the_pending_press(self):
        world = World()
        world.down(D, CTRL | OPT)
        world.listener.rebind(new_hotkey_spec="f13")
        self.assertEqual(world.listener.hotkey, (0, 105))
        self.assertEqual(world.listener.hotkey_state, {"pressed_at": None, "skip_up": False})
        world.listener.rebind(new_hotkey_spec="")
        self.assertIsNone(world.listener.hotkey)
        self.assertEqual(world.listener.binding["hotkey_spec"], "")
        self.assertEqual(world.logs, ["● dictation started via hotkey", "● alternate hotkey is now f13",
                                      "● alternate hotkey off"])

    def test_once_shutdown_is_requested_events_pass_through_untouched(self):
        world = World()
        world.stopping = True
        world.press()
        world.down(A)
        world.down(D, CTRL | OPT)
        world.send(FakeQuartz.kCGEventTapDisabledByTimeout)
        self.assertEqual((world.engine.calls, world.keydowns["count"], world.quartz.enables), ([], 0, []))


if __name__ == "__main__":
    unittest.main()
