"""GestureEngine: the hands-free watchdog and the ordering of capture actions.

Real scrivo.GestureEngine with threading.Timer replaced by a manual timer, so
every deadline fires exactly when the test says and on the thread the test
chooses. No event tap, no microphone, no wall-clock waits.

Findings covered (audit ledger ids): F5 the hands-free watchdog belongs to the
capture, not the key epoch; F6 start / finish / discard run in decision order
and never interleave across threads.
"""

import threading
import unittest
from unittest import mock

import gestures
import scrivo


class ManualTimers:
    """threading.Timer stand-in: arming records (delay, handler, args); the
    test fires what it wants, when it wants."""

    def __init__(self):
        self.armed = []

    def __call__(self, delay, handler, args=()):
        outer = self

        class Timer:
            daemon = False

            def start(self):
                outer.armed.append((delay, handler, tuple(args)))

        return Timer()

    def only(self, delay):
        matching = [t for t in self.armed if t[0] == delay]
        assert len(matching) == 1, matching
        _, handler, args = matching[0]
        return lambda: handler(*args)


class EngineHarness(unittest.TestCase):
    def setUp(self):
        self.timers = ManualTimers()
        for patcher in (mock.patch.object(gestures.threading, "Timer", self.timers),
                        mock.patch.object(gestures, "log", lambda msg: None)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.events = []
        self.engine = scrivo.GestureEngine(lambda: self.events.append("start"),
                                          lambda: self.events.append("finish"),
                                          lambda: self.events.append("discard"))

    def arm_hands_free(self):
        """Two quick taps: the second one's release arms hands-free and its
        watchdog (HANDS_FREE_MAX_S); returns a callable that fires the watchdog."""
        before = len(self.timers.armed)
        self.engine.pressed(); self.engine.released()
        self.engine.pressed(); self.engine.released()
        self.assertEqual(self.engine.snapshot(), (True, True))
        watchdogs = [t for t in self.timers.armed[before:] if t[0] == scrivo.HANDS_FREE_MAX_S]
        self.assertEqual(len(watchdogs), 1)
        _, handler, args = watchdogs[0]
        return lambda: handler(*args)


# -- F5: the watchdog belongs to the capture, not the key epoch ----------------

class HandsFreeWatchdogTests(EngineHarness):
    def test_watchdog_finishes_an_untouched_hands_free_capture(self):
        fire_watchdog = self.arm_hands_free()
        fire_watchdog()
        self.assertEqual(self.events, ["start", "finish"])
        self.assertEqual(self.engine.snapshot(), (False, False))

    def test_watchdog_survives_a_stop_press_whose_release_is_lost(self):
        fire_watchdog = self.arm_hands_free()
        self.engine.pressed()           # the stop tap goes down ... and its key-up never arrives
        self.assertEqual(self.engine.snapshot(), (True, True))
        fire_watchdog()                 # ten minutes later
        self.assertEqual(self.events, ["start", "finish"],
                         "a press disarmed the hands-free watchdog")
        self.assertEqual(self.engine.snapshot(), (False, False))
        self.engine.released()          # the late release is a no-op
        self.assertEqual(self.events, ["start", "finish"])

    def test_menu_armed_hands_free_watchdog_survives_a_press_too(self):
        self.assertTrue(self.engine.force_start())
        fire_watchdog = self.timers.only(scrivo.HANDS_FREE_MAX_S)
        self.engine.pressed()
        fire_watchdog()
        self.assertEqual(self.events, ["start", "finish"])
        self.assertEqual(self.engine.snapshot(), (False, False))

    def test_a_stale_watchdog_cannot_end_a_newer_hands_free_capture(self):
        fire_first = self.arm_hands_free()
        self.engine.pressed(); self.engine.released()        # stop tap: finish
        self.assertEqual(self.events, ["start", "finish"])
        self.events.clear()
        fire_second = self.arm_hands_free()
        fire_first()                                         # the old capture's deadline
        self.assertEqual(self.events, ["start"])
        self.assertEqual(self.engine.snapshot(), (True, True))
        fire_second()
        self.assertEqual(self.events, ["start", "finish"])


# -- F6: actions run in decision order and never interleave across threads ----

class OrderedActionTests(EngineHarness):
    def test_stale_tap_expiry_cannot_discard_a_newer_capture(self):
        capture = scrivo.CaptureService()
        capture._engine = mock.Mock()        # engine already live: begin() does not cold-start
        capture.idle_release_s = -1
        order = []
        discard_entered = threading.Event()
        resume_discard = threading.Event()
        second_start_ran = threading.Event()
        self.addCleanup(resume_discard.set)

        def on_start():
            order.append("start")
            capture.begin()
            if order.count("start") == 2:
                second_start_ran.set()

        def on_discard():
            discard_entered.set()
            resume_discard.wait(5)           # the timer thread is preempted right here
            order.append("discard")
            capture.abort()

        def on_finish():
            order.append("finish")
            capture.end()

        engine = scrivo.GestureEngine(on_start, on_finish, on_discard)
        engine.pressed(); engine.released()  # a lone short tap arms the expiry timer
        self.assertTrue(capture.is_active())
        timer_thread = threading.Thread(target=self.timers.only(scrivo.DOUBLE_TAP_WINDOW_S),
                                        daemon=True)
        timer_thread.start()                 # the window closes: discard decided, lock released
        self.assertTrue(discard_entered.wait(5))
        # The timer thread sits inside on_discard. The user presses again for a
        # real push-to-talk dictation: the decision is immediate, the action
        # must wait its turn, and the tap thread must not be held up.
        engine.pressed()
        self.assertEqual(engine.snapshot(), (True, False))
        self.assertEqual(order.count("start"), 1,
                         "start ran on the tap thread while the stale discard was mid-flight")
        resume_discard.set()
        self.assertTrue(second_start_ran.wait(5))
        timer_thread.join(5)
        self.assertFalse(timer_thread.is_alive())
        self.assertEqual(order, ["start", "discard", "start"])
        self.assertTrue(capture.is_active(), "the stale discard killed the new capture")

    def test_a_release_decided_while_start_is_blocked_is_queued_not_run_concurrently(self):
        order = []
        start_entered = threading.Event()
        resume_start = threading.Event()
        self.addCleanup(resume_start.set)

        def on_start():
            start_entered.set()
            resume_start.wait(5)             # e.g. the capture gate is busy
            order.append("start")

        engine = scrivo.GestureEngine(on_start, lambda: order.append("finish"),
                                     lambda: order.append("discard"))
        tap_thread = threading.Thread(target=engine.pressed, daemon=True)
        tap_thread.start()
        self.assertTrue(start_entered.wait(5))
        with mock.patch.object(gestures, "HOLD_THRESHOLD_S", 0.0):
            engine.released()                # the resync poller's synthesized release
        self.assertEqual(engine.snapshot(), (False, False))   # decided at once ...
        self.assertEqual(order, [], "finish ran while start was still in flight")
        resume_start.set()
        tap_thread.join(5)
        self.assertEqual(order, ["start", "finish"])        # ... acted on in order

    def test_a_menu_finish_queued_behind_a_callback_carries_its_own_action(self):
        """force_finish returns before its finish runs when another thread is
        draining, so caller state set around the call (Windows: copy, don't
        insert) is gone by then. The decision has to travel with the action."""
        order = []
        start_entered = threading.Event()
        resume_start = threading.Event()
        self.addCleanup(resume_start.set)

        def on_start():
            start_entered.set()
            resume_start.wait(5)             # e.g. the mic is slow to open
            order.append("start")

        engine = scrivo.GestureEngine(on_start, lambda: order.append("finish"),
                                     lambda: order.append("discard"))
        tap_thread = threading.Thread(target=engine.pressed, daemon=True)
        tap_thread.start()
        self.assertTrue(start_entered.wait(5))
        self.assertTrue(engine.force_finish(finish=lambda: order.append("menu finish")))
        self.assertEqual(order, [], "the queued finish ran while start was in flight")
        resume_start.set()
        tap_thread.join(5)
        self.assertEqual(order, ["start", "menu finish"])
        self.assertEqual(engine.snapshot(), (False, False))

    def test_force_finish_from_inside_a_callback_queues_instead_of_waiting(self):
        """on_mic_failed calls force_finish from inside on_start, on the
        draining thread: it must return at once (waiting for its own finish
        there would deadlock) and the finish runs right after the start."""
        order = []
        results = []

        def on_start():
            results.append(engine.force_finish())
            order.append("start")

        engine = scrivo.GestureEngine(on_start, lambda: order.append("finish"),
                                     lambda: order.append("discard"))
        tap_thread = threading.Thread(target=engine.pressed, daemon=True)
        tap_thread.start()
        tap_thread.join(5)
        self.assertFalse(tap_thread.is_alive(), "force_finish waited on its own drainer")
        self.assertEqual(results, [True])
        self.assertEqual(order, ["start", "finish"])


# -- #19 review: a stale mic failure never ends a newer gesture -----------------

class EpochBoundFinishTests(EngineHarness):
    def test_a_finish_bound_to_an_older_epoch_leaves_the_gesture_alone(self):
        self.arm_hands_free()
        stale = self.engine.epoch()     # the capture whose open later fails began here
        self.engine.pressed()           # a newer key decision before the failure lands
        self.assertFalse(self.engine.force_finish(if_epoch=stale))
        self.assertEqual(self.engine.snapshot()[0], True, "the newer gesture keeps recording")
        self.assertEqual(self.events, ["start"])

    def test_a_finish_bound_to_the_current_epoch_still_ends_it(self):
        self.arm_hands_free()
        self.assertTrue(self.engine.force_finish(if_epoch=self.engine.epoch()))
        self.assertEqual(self.engine.snapshot(), (False, False))
        self.assertEqual(self.events, ["start", "finish"])


# -- characterization: the decisions pinned before gestures.py -----------------

class GestureDecisionTests(EngineHarness):
    def test_a_lone_tap_waits_out_the_double_tap_window_then_is_discarded(self):
        self.engine.pressed(); self.engine.released()
        self.assertEqual(self.engine.snapshot(), (True, False))   # recording until the verdict
        self.assertEqual(self.events, ["start"])
        self.timers.only(scrivo.DOUBLE_TAP_WINDOW_S)()
        self.assertEqual(self.events, ["start", "discard"])
        self.assertEqual(self.engine.snapshot(), (False, False))

    def test_a_held_press_is_push_to_talk(self):
        with mock.patch.object(gestures, "HOLD_THRESHOLD_S", 0.0):
            self.engine.pressed(); self.engine.released()
        self.assertEqual(self.events, ["start", "finish"])
        self.assertEqual(self.engine.snapshot(), (False, False))
        self.assertEqual(self.timers.armed, [])                   # no tap verdict pending

    def test_a_chord_drops_push_to_talk_but_not_hands_free(self):
        self.engine.pressed()
        self.engine.chorded()
        self.assertEqual(self.events, ["start", "discard"])
        self.assertEqual(self.engine.snapshot(), (False, False))
        self.events.clear()
        self.arm_hands_free()
        self.engine.chorded()                                     # the trigger is no longer held
        self.assertEqual(self.events, ["start"])
        self.assertEqual(self.engine.snapshot(), (True, True))

    def test_a_menu_start_is_refused_while_recording_and_arms_hands_free_otherwise(self):
        hands_free = []
        engine = scrivo.GestureEngine(lambda: self.events.append("start"),
                                     lambda: self.events.append("finish"),
                                     lambda: self.events.append("discard"),
                                     on_hands_free=lambda: hands_free.append(True))
        engine.pressed()
        self.assertFalse(engine.force_start())
        engine.force_reset()
        self.assertEqual(self.events, ["start", "discard"])
        self.assertTrue(engine.force_start())
        self.assertEqual(self.events, ["start", "discard", "start"])
        self.assertEqual(hands_free, [True])
        self.assertEqual(engine.snapshot(), (True, True))

    def test_force_finish_and_force_reset_report_and_end_only_a_live_recording(self):
        self.assertFalse(self.engine.force_finish())
        self.engine.force_reset()
        self.assertEqual(self.events, [])
        self.engine.pressed()
        self.assertTrue(self.engine.force_finish())
        self.assertEqual(self.events, ["start", "finish"])

    def test_the_epoch_moves_on_every_press_release_verdict_and_forced_end(self):
        epochs = [self.engine.epoch()]
        self.engine.pressed(); epochs.append(self.engine.epoch())
        self.engine.released(); epochs.append(self.engine.epoch())   # a tap: its verdict timer
        self.engine.force_finish(); epochs.append(self.engine.epoch())
        self.engine.force_reset(); epochs.append(self.engine.epoch())
        self.assertEqual(epochs, sorted(set(epochs)))

    def test_a_callback_that_raises_is_logged_and_leaves_the_gesture_idle(self):
        logs = []

        def failing_start():
            raise RuntimeError("mic busy")
        engine = scrivo.GestureEngine(failing_start, lambda: self.events.append("finish"),
                                     lambda: self.events.append("discard"))
        with mock.patch.object(gestures, "log", logs.append):
            engine.pressed()
        self.assertEqual(engine.snapshot(), (False, False))
        self.assertEqual(logs, ["! gesture callback failed: mic busy"])
        engine.released()                                        # nothing left to finish
        self.assertEqual(self.events, [])

    def test_arming_hands_free_logs_the_hint(self):
        logs = []
        with mock.patch.object(gestures, "log", logs.append):
            self.arm_hands_free()
        self.assertEqual(logs, ["● hands-free (tap again to stop)"])

    def test_without_a_warning_callback_only_the_watchdog_is_armed(self):
        self.arm_hands_free()
        warn_at = gestures.HANDS_FREE_MAX_S - gestures.HANDS_FREE_WARN_S
        self.assertEqual([t for t in self.timers.armed if t[0] == warn_at], [])


# -- the hands-free limit gives a heads-up before it ends the recording ---------

class HandsFreeWarningTests(EngineHarness):
    """2026-11-04: the 600 s watchdog cut a real 10-minute dictation mid-word
    with no warning. The limit is longer now and warns HANDS_FREE_WARN_S first."""

    def setUp(self):
        super().setUp()
        self.engine = scrivo.GestureEngine(lambda: self.events.append("start"),
                                          lambda: self.events.append("finish"),
                                          lambda: self.events.append("discard"),
                                          on_hands_free_warning=self.warned)
        self.warn_at = gestures.HANDS_FREE_MAX_S - gestures.HANDS_FREE_WARN_S
        self.tokens = []

    def warned(self, token):
        self.tokens.append(token)
        self.events.append("warning")

    def test_the_limit_outlasts_a_ten_minute_dictation(self):
        self.assertGreaterEqual(gestures.HANDS_FREE_MAX_S, 1800.0)
        self.assertGreater(self.warn_at, 600.0)

    def test_the_warning_comes_before_the_watchdog_and_keeps_recording(self):
        fire_watchdog = self.arm_hands_free()
        self.timers.only(self.warn_at)()
        self.assertEqual(self.events, ["start", "warning"])
        self.assertEqual(self.engine.snapshot(), (True, True), "a warning never ends the recording")
        fire_watchdog()
        self.assertEqual(self.events, ["start", "warning", "finish"])

    def test_no_warning_after_the_user_stopped(self):
        self.arm_hands_free()
        self.engine.pressed(); self.engine.released()            # the stop tap
        self.timers.only(self.warn_at)()
        self.assertEqual(self.events, ["start", "finish"])

    def test_an_old_capture_s_warning_does_not_speak_for_a_new_one(self):
        self.arm_hands_free()
        old_warning = self.timers.only(self.warn_at)
        self.engine.pressed(); self.engine.released()            # stop
        self.timers.armed.clear()
        self.arm_hands_free()                                    # a new hands-free capture
        old_warning()
        self.assertEqual(self.events, ["start", "finish", "start"])
        self.timers.only(self.warn_at)()
        self.assertEqual(self.events[-1], "warning")

    def test_a_menu_started_recording_warns_too(self):
        self.assertTrue(self.engine.force_start())
        self.timers.only(self.warn_at)()
        self.assertEqual(self.events, ["start", "warning"])

    def test_the_token_it_carries_is_current_only_for_its_own_capture(self):
        # A warning queued for the main thread re-checks this there, so one
        # that waited while its capture ended never shows on the next one.
        self.arm_hands_free()
        self.timers.only(self.warn_at)()
        (token,) = self.tokens
        self.assertTrue(self.engine.is_current_hands_free(token))
        self.engine.pressed(); self.engine.released()            # stop
        self.assertFalse(self.engine.is_current_hands_free(token))
        self.arm_hands_free()                                    # a newer capture
        self.assertFalse(self.engine.is_current_hands_free(token))

    def test_a_failing_warning_leaves_the_recording_and_its_watchdog_alone(self):
        def failing(_token):
            raise RuntimeError("no main thread")
        engine = scrivo.GestureEngine(lambda: self.events.append("start"),
                                     lambda: self.events.append("finish"),
                                     lambda: self.events.append("discard"),
                                     on_hands_free_warning=failing)
        engine.pressed(); engine.released()
        engine.pressed(); engine.released()
        logs = []
        with mock.patch.object(gestures, "log", logs.append):
            self.timers.only(self.warn_at)()
        self.assertEqual(engine.snapshot(), (True, True), "the mic is still recording")
        self.assertIn("! hands-free warning failed: no main thread", logs)
        self.timers.only(gestures.HANDS_FREE_MAX_S)()
        self.assertEqual(self.events, ["start", "finish"])


if __name__ == "__main__":
    unittest.main()
