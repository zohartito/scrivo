"""The lifecycle pieces the Mac and Windows apps share: the shutdown boundary,
the capture gate, which reason holds the gate, and the in-flight check the
drains poll. Pinned before they moved out of scrivo.py; runs on both
platforms (no AppKit, no microphone, no signal is delivered).
"""
import queue
import signal
import threading
import types
import unittest

import lifecycle
import scrivo


class ShutdownBoundaryTests(unittest.TestCase):
    def test_a_teardown_wait_is_its_cap_unless_sigterm_set_a_shared_deadline(self):
        boundary = scrivo.ShutdownBoundary()
        self.assertEqual(boundary.teardown_wait_s(2.0, 100.0), 2.0)
        boundary.request_stop(signal.SIGINT)                  # Ctrl-C: no launchd deadline
        self.assertEqual(boundary.teardown_wait_s(2.0, 1e9), 2.0)

        sigterm = scrivo.ShutdownBoundary()
        sigterm.request_stop(signal.SIGTERM)
        at = sigterm.stop_at
        keep_by = at + scrivo.LAUNCHD_EXIT_TIMEOUT_S - scrivo.SIGTERM_KEEP_RESERVE_S
        self.assertEqual(sigterm.teardown_wait_s(10.0, at), keep_by - at)
        self.assertEqual(sigterm.teardown_wait_s(1.0, at), 1.0)
        self.assertEqual(sigterm.teardown_wait_s(10.0, keep_by + 1.0), 0.0)

    def test_the_first_stop_request_drains_and_a_second_one_stops_at_once(self):
        boundary = scrivo.ShutdownBoundary()
        boundary.request_stop(signal.SIGTERM)
        self.assertTrue(boundary.stop_event.is_set())
        self.assertFalse(boundary.requested())
        self.assertEqual(boundary.stop_signal, signal.SIGTERM)
        boundary.request_stop(signal.SIGINT)
        self.assertTrue(boundary.requested())
        self.assertEqual(boundary.stop_signal, signal.SIGTERM)  # the first signal is the one recorded

    def test_enqueue_queues_only_while_live_and_hands_refused_live_jobs_to_keep(self):
        boundary, jobs, kept = scrivo.ShutdownBoundary(), queue.Queue(), []
        self.assertTrue(boundary.enqueue(jobs, ("live", 1), keep=kept.append))
        boundary.request()
        self.assertFalse(boundary.enqueue(jobs, ("live", 2), keep=kept.append))
        self.assertFalse(boundary.enqueue(jobs, ("retry", 3), keep=kept.append))
        self.assertEqual(kept, [("live", 2)])
        self.assertEqual(jobs.qsize(), 1)

    def test_discard_queued_keeps_live_jobs_and_balances_the_task_count(self):
        jobs, kept = queue.Queue(), []
        for job in (("live", "a"), ("retry", "b"), ("live", "c")):
            jobs.put(job)
        self.assertEqual(scrivo.ShutdownBoundary.discard_queued(jobs, keep=kept.append), 3)
        self.assertEqual(kept, [("live", "a"), ("live", "c")])
        self.assertEqual(jobs.unfinished_tasks, 0)
        self.assertEqual(scrivo.ShutdownBoundary.discard_queued(jobs), 0)

    def test_stop_capture_aborts_then_shuts_the_capture_down_even_if_abort_raises(self):
        calls = []
        closable = types.SimpleNamespace(abort=lambda: calls.append("abort"),
                                         shutdown=lambda: calls.append("shutdown"))
        boundary = scrivo.ShutdownBoundary()
        boundary.stop_capture(closable)
        self.assertTrue(boundary.requested())
        legacy = types.SimpleNamespace(abort=lambda: calls.append("abort"),
                                       release_soon=lambda: calls.append("release_soon"))
        scrivo.ShutdownBoundary().stop_capture(legacy)

        def broken_abort():
            raise RuntimeError("engine gone")
        failing = types.SimpleNamespace(abort=broken_abort, shutdown=lambda: calls.append("shutdown"))
        with self.assertRaises(RuntimeError):
            scrivo.ShutdownBoundary().stop_capture(failing)
        self.assertEqual(calls, ["abort", "shutdown", "abort", "release_soon", "shutdown"])


class LifecycleGateTests(unittest.TestCase):
    def make(self):
        shutdown, gate = scrivo.ShutdownBoundary(), scrivo.CaptureGate()
        lifecycle = scrivo.Lifecycle(shutdown, gate, end_recording=lambda: None, busy=lambda: False)
        return shutdown, gate, lifecycle

    @staticmethod
    def gate_open(gate) -> bool:
        with gate.starting() as allowed:
            return allowed

    def test_one_reason_holds_the_gate_and_only_that_reason_reopens_it(self):
        _shutdown, gate, lifecycle = self.make()
        self.assertTrue(lifecycle.close("updating"))
        self.assertFalse(self.gate_open(gate))
        self.assertFalse(lifecycle.close("restarting"))
        self.assertEqual(lifecycle.closed, "updating")
        lifecycle.reopen("restarting")                        # not the holder: no change
        self.assertFalse(self.gate_open(gate))
        lifecycle.reopen("updating")
        self.assertIsNone(lifecycle.closed)
        self.assertTrue(self.gate_open(gate))

    def test_run_if_open_runs_the_action_only_while_the_gate_is_open(self):
        shutdown, _gate, lifecycle = self.make()
        ran = []
        self.assertIsNone(lifecycle.run_if_open(lambda: ran.append("retry")))
        lifecycle.close("switching engines")
        self.assertEqual(lifecycle.run_if_open(lambda: ran.append("late")), "switching engines")
        shutdown.request()
        self.assertEqual(lifecycle.run_if_open(lambda: ran.append("later")), "stopping")
        self.assertEqual(ran, ["retry"])

    def test_nothing_closes_or_reopens_once_stopping(self):
        shutdown, gate, lifecycle = self.make()
        lifecycle.close("updating")
        shutdown.request()
        lifecycle.reopen("updating")
        self.assertFalse(self.gate_open(gate))
        self.assertFalse(lifecycle.close("restarting"))
        self.assertFalse(lifecycle.quit())


class InFlightTests(unittest.TestCase):
    def test_recording_queued_jobs_and_pending_deliveries_are_in_flight_but_one_warmup_is_not(self):
        recording = {"active": False}
        capture = types.SimpleNamespace(is_active=lambda: recording["active"])
        jobs, deliveries = queue.Queue(), scrivo.PendingDeliveries()
        self.assertFalse(scrivo.dictation_in_flight(capture, jobs, deliveries))
        recording["active"] = True
        self.assertTrue(scrivo.dictation_in_flight(capture, jobs, deliveries))
        recording["active"] = False
        jobs.put(("warmup",))
        self.assertTrue(scrivo.dictation_in_flight(capture, jobs, deliveries))
        self.assertFalse(scrivo.dictation_in_flight(capture, jobs, deliveries, warming=True))
        jobs.put(("live",))
        self.assertTrue(scrivo.dictation_in_flight(capture, jobs, deliveries, warming=True))
        while not jobs.empty():
            jobs.get_nowait()
            jobs.task_done()
        deliveries.add()
        self.assertTrue(scrivo.dictation_in_flight(capture, jobs, deliveries))



class DrainHelperTests(unittest.TestCase):
    """The drain pieces Windows' Quit and Restart reuse (restructure step 7)."""

    def test_a_drain_polls_until_idle_or_gives_up_at_its_deadline(self):
        now = [0.0]

        def sleep(seconds):
            now[0] += seconds
        self.assertTrue(scrivo.wait_until_idle(lambda: now[0] < 0.6, poll_s=0.25, deadline_s=1.0,
                                              clock=lambda: now[0], sleep=sleep))
        self.assertEqual(now[0], 0.75)
        now[0] = 0.0
        self.assertFalse(scrivo.wait_until_idle(lambda: True, poll_s=0.25, deadline_s=1.0,
                                               clock=lambda: now[0], sleep=sleep))
        self.assertEqual(now[0], 1.0)

    def test_ending_a_gesture_runs_the_finish_it_was_handed(self):
        # Windows hands a finish that copies instead of inserting; the Mac
        # hands the engine's own on_finish.
        from gestures import GestureEngine
        engine_finishes, handed = [], []
        engine = GestureEngine(lambda: None, lambda: engine_finishes.append(1), lambda: None)
        engine.force_start()
        capture = types.SimpleNamespace(is_active=lambda: True)
        self.assertEqual(scrivo.end_capture_now(engine, capture, lambda: handed.append("copy")), "gesture")
        self.assertEqual((handed, engine_finishes), (["copy"], []))
        self.assertEqual(scrivo.end_capture_now(engine, capture, lambda: handed.append("orphan")), "orphan")
        self.assertEqual(handed, ["copy", "orphan"])


class ExitGuardTests(unittest.TestCase):
    """2026-10-19: a shutdown that began and never finished left Scrivo alive for
    15+ hours: tap disabled, signal timers stopped, SIGTERM ignored, nothing
    logged. The guard ends such a process."""

    def guard(self, *, finished=False, restart=False, limit=45.0):
        event = types.SimpleNamespace(wait=lambda seconds: (waited.append(seconds), finished)[1])
        waited, logs, exits = [], [], []
        lifecycle.exit_if_shutdown_stuck(event, lambda: restart, log=logs.append,
                                         exit_process=exits.append, limit=limit)
        return waited, logs, exits

    def test_a_run_that_returns_in_time_is_left_alone(self):
        waited, logs, exits = self.guard(finished=True)
        self.assertEqual((waited, logs, exits), ([45.0], [], []))

    def test_a_stuck_quit_logs_and_exits_zero(self):
        waited, logs, exits = self.guard()
        self.assertEqual(exits, [0])
        self.assertEqual(logs, ["! shutdown did not finish within 45s — exiting"])

    def test_a_stuck_restart_exits_nonzero_so_keepalive_starts_scrivo_again(self):
        _waited, logs, exits = self.guard(restart=True)
        self.assertEqual(exits, [lifecycle.RESTART_FAILED_EXIT])
        self.assertIn("so Scrivo starts again", logs[0])

    def test_it_stands_down_the_moment_run_finishes(self):
        finished, exits = threading.Event(), []
        thread = threading.Thread(target=lifecycle.exit_if_shutdown_stuck, args=(finished, lambda: False),
                                  kwargs={"exit_process": exits.append, "limit": 30.0}, daemon=True)
        thread.start()
        finished.set()
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(exits, [])

    def test_a_blocked_log_write_cannot_keep_it_from_exiting(self):
        release, exits = threading.Event(), []
        try:
            lifecycle.exit_if_shutdown_stuck(types.SimpleNamespace(wait=lambda seconds: False), lambda: False,
                                             log=lambda line: release.wait(30), exit_process=exits.append,
                                             limit=1.0)
            self.assertEqual(exits, [0])
        finally:
            release.set()

    def test_the_limit_leaves_room_for_the_slowest_legitimate_teardown(self):
        slowest = (lifecycle.RESTART_DRAIN_DEADLINE_S + lifecycle.APP_DRAIN_TIMEOUT
                   + lifecycle.RESTART_RELEASE_WAIT_S + lifecycle.SIGTERM_RESTORE_WAIT_S)
        self.assertGreater(lifecycle.EXIT_GUARD_S, slowest + 10)


if __name__ == "__main__":
    unittest.main()
