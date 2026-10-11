"""The lifecycle pieces the Mac and Windows apps share: the shutdown boundary,
the capture gate, which reason holds the gate, and the in-flight check the
drains poll. Pinned before they moved out of sotto.py; runs on both
platforms (no AppKit, no microphone, no signal is delivered).
"""
import queue
import signal
import types
import unittest

import sotto


class ShutdownBoundaryTests(unittest.TestCase):
    def test_a_teardown_wait_is_its_cap_unless_sigterm_set_a_shared_deadline(self):
        boundary = sotto.ShutdownBoundary()
        self.assertEqual(boundary.teardown_wait_s(2.0, 100.0), 2.0)
        boundary.request_stop(signal.SIGINT)                  # Ctrl-C: no launchd deadline
        self.assertEqual(boundary.teardown_wait_s(2.0, 1e9), 2.0)

        sigterm = sotto.ShutdownBoundary()
        sigterm.request_stop(signal.SIGTERM)
        at = sigterm.stop_at
        keep_by = at + sotto.LAUNCHD_EXIT_TIMEOUT_S - sotto.SIGTERM_KEEP_RESERVE_S
        self.assertEqual(sigterm.teardown_wait_s(10.0, at), keep_by - at)
        self.assertEqual(sigterm.teardown_wait_s(1.0, at), 1.0)
        self.assertEqual(sigterm.teardown_wait_s(10.0, keep_by + 1.0), 0.0)

    def test_the_first_stop_request_drains_and_a_second_one_stops_at_once(self):
        boundary = sotto.ShutdownBoundary()
        boundary.request_stop(signal.SIGTERM)
        self.assertTrue(boundary.stop_event.is_set())
        self.assertFalse(boundary.requested())
        self.assertEqual(boundary.stop_signal, signal.SIGTERM)
        boundary.request_stop(signal.SIGINT)
        self.assertTrue(boundary.requested())
        self.assertEqual(boundary.stop_signal, signal.SIGTERM)  # the first signal is the one recorded

    def test_enqueue_queues_only_while_live_and_hands_refused_live_jobs_to_keep(self):
        boundary, jobs, kept = sotto.ShutdownBoundary(), queue.Queue(), []
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
        self.assertEqual(sotto.ShutdownBoundary.discard_queued(jobs, keep=kept.append), 3)
        self.assertEqual(kept, [("live", "a"), ("live", "c")])
        self.assertEqual(jobs.unfinished_tasks, 0)
        self.assertEqual(sotto.ShutdownBoundary.discard_queued(jobs), 0)

    def test_stop_capture_aborts_then_shuts_the_capture_down_even_if_abort_raises(self):
        calls = []
        closable = types.SimpleNamespace(abort=lambda: calls.append("abort"),
                                         shutdown=lambda: calls.append("shutdown"))
        boundary = sotto.ShutdownBoundary()
        boundary.stop_capture(closable)
        self.assertTrue(boundary.requested())
        legacy = types.SimpleNamespace(abort=lambda: calls.append("abort"),
                                       release_soon=lambda: calls.append("release_soon"))
        sotto.ShutdownBoundary().stop_capture(legacy)

        def broken_abort():
            raise RuntimeError("engine gone")
        failing = types.SimpleNamespace(abort=broken_abort, shutdown=lambda: calls.append("shutdown"))
        with self.assertRaises(RuntimeError):
            sotto.ShutdownBoundary().stop_capture(failing)
        self.assertEqual(calls, ["abort", "shutdown", "abort", "release_soon", "shutdown"])


class LifecycleGateTests(unittest.TestCase):
    def make(self):
        shutdown, gate = sotto.ShutdownBoundary(), sotto.CaptureGate()
        lifecycle = sotto.Lifecycle(shutdown, gate, end_recording=lambda: None, busy=lambda: False)
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
        jobs, deliveries = queue.Queue(), sotto.PendingDeliveries()
        self.assertFalse(sotto.dictation_in_flight(capture, jobs, deliveries))
        recording["active"] = True
        self.assertTrue(sotto.dictation_in_flight(capture, jobs, deliveries))
        recording["active"] = False
        jobs.put(("warmup",))
        self.assertTrue(sotto.dictation_in_flight(capture, jobs, deliveries))
        self.assertFalse(sotto.dictation_in_flight(capture, jobs, deliveries, warming=True))
        jobs.put(("live",))
        self.assertTrue(sotto.dictation_in_flight(capture, jobs, deliveries, warming=True))
        while not jobs.empty():
            jobs.get_nowait()
            jobs.task_done()
        deliveries.add()
        self.assertTrue(sotto.dictation_in_flight(capture, jobs, deliveries))


if __name__ == "__main__":
    unittest.main()
