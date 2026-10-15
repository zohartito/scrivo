"""Onboarding seams; no microphone, model download or app injection."""
import inspect
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

import lifecycle
import scrivo
import scrivo_log
from speech_config import MODEL_PROFILES, MODEL_REVISIONS

ROOT = Path(__file__).resolve().parents[1]


class _Immediate:
    """threading.Thread stand-in that runs the target synchronously on start()."""

    def __init__(self, target, daemon=None):
        self.target = target
        self.daemon = daemon

    def start(self):
        self.target()


def controller(*, managed=False, sealed=False, argv=('scrivo.py', 'run')):
    with patch.object(lifecycle, 'launchd_owns_this_process', return_value=managed), \
         patch.object(lifecycle, 'launched_as_sealed_job', return_value=sealed), \
         patch.object(sys, 'argv', list(argv)):
        return scrivo.RestartController(scrivo.ShutdownBoundary())


class RestartTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'darwin', 'launchd/launchctl restart is macOS-only')
    def test_another_installed_process_never_owns_a_foreground_run(self):
        for pid, expected in ((os.getpid(), True), (os.getpid()+1, False)):
            result = subprocess.CompletedProcess([], 0, stdout=f'\tpid = {pid}\n')
            with patch('subprocess.run', return_value=result):
                self.assertEqual(scrivo.launchd_owns_this_process(), expected)
        with patch('subprocess.run', side_effect=OSError):
            self.assertFalse(scrivo.launchd_owns_this_process())

    def test_sealed_job_is_recognised_from_launchd_env_or_release_layout(self):
        sealed_entry = '/Users/x/Library/Application Support/sotto/releases/' + 'a' * 64 + '/source/scrivo.py'
        cases = (({'XPC_SERVICE_NAME': scrivo.LAUNCHD_LABEL}, 'scrivo.py', True),
                 ({'XPC_SERVICE_NAME': '0'}, sealed_entry, True),
                 ({'XPC_SERVICE_NAME': 'application.com.apple.Terminal.1'}, 'scrivo.py', False))
        for env, entry, expected in cases:
            with patch.dict(os.environ, env), patch.object(sys, 'argv', [entry, 'run']):
                self.assertEqual(scrivo.launched_as_sealed_job(), expected, (env, entry))

    def test_unconfirmed_sealed_job_fails_closed_instead_of_exec(self):
        restart = controller(managed=False, sealed=True)
        with patch('subprocess.run') as launchctl, patch('os.execv') as execute:
            with self.assertRaisesRegex(RuntimeError, 'could not confirm its launchd job'):
                restart.request()
            restart.exec_foreground()
        self.assertFalse(restart.shutdown.requested())
        self.assertFalse(restart.foreground_pending)
        execute.assert_not_called()
        launchctl.assert_not_called()

    def test_foreground_restart_requests_drain_then_execs_same_interpreter_and_args(self):
        argv = ['scrivo.py', 'run', '--trigger', 'right-ctrl', '--hotkey', '']
        restart = controller(argv=argv)
        with patch('subprocess.run') as launchctl, patch('os.execv') as execute:
            self.assertTrue(restart.request())
            self.assertTrue(restart.shutdown.requested())
            self.assertTrue(restart.foreground_pending)
            execute.assert_not_called()
            restart.exec_foreground()
            execute.assert_called_once_with(sys.executable, [sys.executable, *argv])
            launchctl.assert_not_called()

    @unittest.skipUnless(sys.platform == 'darwin', 'Scrivo.app is macOS-only')
    def test_scrivo_app_restarts_as_a_fresh_process_never_in_place(self):
        # Scrivo.app, and the old Sotto.app when this code runs from it (its
        # restart opens that app again, whose start then hands off to Scrivo.app).
        for executable, bundle in (('/Applications/Scrivo.app/Contents/MacOS/Scrivo', '/Applications/Scrivo.app'),
                                   ('/Applications/Sotto.app/Contents/MacOS/Sotto', '/Applications/Sotto.app')):
            with self.subTest(bundle=bundle):
                restart = controller()
                env = {'SCRIVO_LAUNCHER': 'app', 'SCRIVO_APP_EXECUTABLE': executable}
                with patch.dict(os.environ, env), patch('subprocess.Popen') as spawn, patch('os.execv') as execute:
                    self.assertTrue(restart.request())
                    restart.exec_foreground()
                execute.assert_not_called()
                argv = spawn.call_args.args[0]
                self.assertEqual(argv[:3], ['/bin/sh', '-c', scrivo.APP_RELAUNCH_SCRIPT])
                self.assertEqual(argv[4:], [str(os.getppid()), 'io.github.zohartito.scrivo', bundle])
                self.assertTrue(spawn.call_args.kwargs['start_new_session'])

    @unittest.skipUnless(sys.platform == 'darwin', 'Scrivo.app relaunch helper is macOS-only')
    def test_relaunch_helper_waits_for_the_launcher_then_opens_unless_the_login_item_starts(self):
        for kickstart, expected in (('false', 'opened /B/Scrivo.app'), ('true', '')):
            script = (scrivo.APP_RELAUNCH_SCRIPT.replace('/bin/launchctl kickstart', kickstart)
                      .replace('/usr/bin/open -n', 'echo opened'))
            launcher = subprocess.Popen(['/bin/sleep', '0.5'])
            threading.Thread(target=launcher.wait, daemon=True).start()  # reap it, as launchd would
            started = time.monotonic()
            result = subprocess.run(['/bin/sh', '-c', script, 'x', str(launcher.pid), 'label', '/B/Scrivo.app'],
                                    capture_output=True, text=True, timeout=10)
            self.assertGreaterEqual(time.monotonic() - started, 0.4)
            self.assertEqual(result.stdout.strip(), expected)

    def test_supervised_restart_never_blocks_the_caller_on_launchctl(self):
        restart = controller(managed=True)
        created = []
        with patch('threading.Thread', side_effect=lambda **kwargs: created.append(kwargs) or Mock()), \
             patch('subprocess.run') as launchctl:
            self.assertTrue(restart.request())
        launchctl.assert_not_called()
        self.assertTrue(created and created[0]['daemon'])
        self.assertFalse(restart.shutdown.requested())

    @unittest.skipUnless(sys.platform == 'darwin', 'launchd/launchctl restart is macOS-only')
    def test_supervised_restart_reports_a_launchctl_failure_once(self):
        restart = controller(managed=True)
        failures = []
        with patch('threading.Thread', _Immediate), \
             patch('subprocess.run', return_value=subprocess.CompletedProcess([], 1)) as launchctl:
            self.assertTrue(restart.request(on_failure=failures.append))
            self.assertIn('kickstart', launchctl.call_args.args[0])
        self.assertEqual(len(failures), 1)
        self.assertIn('could not restart', failures[0])
        restart.failed('again')
        self.assertEqual(len(failures), 1)
        self.assertFalse(restart.foreground_pending)
        self.assertFalse(restart.shutdown.requested())
        with patch('threading.Thread', _Immediate), \
             patch('subprocess.run', return_value=subprocess.CompletedProcess([], 0)):
            self.assertTrue(restart.request(on_failure=failures.append))
        self.assertEqual(len(failures), 1)

    def test_failed_engine_restart_restores_previous_setting(self):
        with patch('speech_config.load_engine_mode', return_value='whisper'), \
                patch('speech_config.save_engine_mode') as save:
            with self.assertRaises(RuntimeError):
                scrivo.persist_engine_and_restart('nemotron', lambda **_: False)
            self.assertEqual([c.args[0] for c in save.call_args_list], ['nemotron', 'whisper'])

    def test_late_restart_failure_restores_previous_engine_and_reports(self):
        restart = controller()
        reported = []
        with patch('speech_config.load_engine_mode', return_value='whisper'), \
             patch('speech_config.save_engine_mode') as save:
            scrivo.persist_engine_and_restart('nemotron', restart.request, on_failure=reported.append)
            self.assertEqual([c.args[0] for c in save.call_args_list], ['nemotron'])
            restart.failed('exec failed')  # e.g. os.execv raised during teardown
            self.assertEqual([c.args[0] for c in save.call_args_list], ['nemotron', 'whisper'])
        self.assertEqual(reported, ['exec failed'])

    def test_dead_route_uses_its_own_runtime_restart_callback(self):
        service = scrivo.CaptureService()
        # Old enough to restart, whatever the machine's uptime (CI runners boot minutes before).
        service._process_started_at = time.monotonic() - scrivo.SILENCE_RESTART_AFTER_S - 1
        service.restart_callback = Mock(return_value=True)
        with patch('subprocess.run') as external:
            self.assertTrue(service._restart_app())
            external.assert_not_called()
        service.restart_callback.assert_called_once_with()

    def test_dead_route_restart_refusal_keeps_the_rung_armed(self):
        service = scrivo.CaptureService()
        service._process_started_at = time.monotonic() - scrivo.SILENCE_RESTART_AFTER_S - 1
        service.restart_callback = controller(sealed=True).request
        self.assertFalse(service._restart_app())


class ChordTests(unittest.TestCase):
    """A key pressed while the trigger is held is a shortcut, not dictation."""

    def engine(self):
        events = []
        engine = scrivo.GestureEngine(lambda: events.append('start'), lambda: events.append('finish'),
                                     lambda: events.append('discard'))
        return engine, events

    def test_chord_during_push_to_talk_discards_and_release_does_not_paste(self):
        engine, events = self.engine()
        engine.pressed()
        time.sleep(scrivo.HOLD_THRESHOLD_S + 0.05)
        engine.chorded()
        engine.released()
        self.assertEqual(events, ['start', 'discard'])
        self.assertEqual(engine.snapshot(), (False, False))

    def test_chord_while_idle_or_hands_free_changes_nothing(self):
        engine, events = self.engine()
        engine.chorded()
        self.assertEqual(events, [])
        self.assertTrue(engine.force_start())  # menu-armed hands-free
        engine.chorded()
        self.assertEqual(events, ['start'])
        self.assertEqual(engine.snapshot(), (True, True))
        self.assertTrue(engine.force_finish())
        self.assertEqual(events, ['start', 'finish'])


class GlossarySourceTests(unittest.TestCase):
    def test_explicit_file_wins_else_the_data_folder_glossary_if_present(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            with patch('scrivo_paths.DATA_DIR', data):
                self.assertEqual(scrivo.glossary_source('/x/terms.txt'), Path('/x/terms.txt'))
                self.assertIsNone(scrivo.glossary_source(None))
                (data / 'glossary.txt').write_text('Scrivo\n', encoding='utf-8')
                self.assertEqual(scrivo.glossary_source(None), data / 'glossary.txt')


class HandsFreeHintTests(unittest.TestCase):
    """Hands-free keeps recording after the key is up, so the overlay says how to stop."""

    def test_hint_names_the_key_to_tap(self):
        self.assertEqual(scrivo.hands_free_hint('right-cmd'), 'tap right ⌘ to stop')
        self.assertEqual(scrivo.hands_free_hint('left-option'), 'tap left ⌥ to stop')
        self.assertEqual(scrivo.hands_free_hint('fn'), 'tap fn to stop')

    def test_double_tap_and_menu_start_announce_hands_free_once(self):
        events = []
        engine = scrivo.GestureEngine(lambda: events.append('start'), lambda: events.append('finish'),
                                     lambda: events.append('discard'),
                                     on_hands_free=lambda: events.append('hands-free'))
        engine.pressed(); engine.released()
        engine.pressed(); engine.released()          # quick second tap = hands-free
        self.assertEqual(engine.snapshot(), (True, True))
        self.assertEqual(events.count('hands-free'), 1)
        engine.pressed(); engine.released()          # one tap stops it
        self.assertEqual(events[-1], 'finish')
        events.clear()
        self.assertTrue(engine.force_start())
        self.assertEqual(events, ['start', 'hands-free'])

    @unittest.skipUnless(sys.platform == 'darwin', 'AppKit overlay')
    def test_overlay_widens_for_the_hint_and_returns_to_the_plain_pill(self):
        import ui
        status = ui.StatusUI.__new__(ui.StatusUI)
        status._panel = status._make_island()
        status._orb = status._make_orb()
        status._hint = None
        self.assertEqual(status._layout_island(), ui.ISLAND_W)
        status._hint = 'tap right ⌘ to stop'
        width = status._layout_island()
        self.assertGreater(width, ui.ISLAND_W + 60)
        self.assertFalse(status._hint_layer.isHidden())
        self.assertEqual(status._hint_layer.string(), 'tap right ⌘ to stop')
        self.assertLess(status._orb.position().x, ui.ISLAND_W / 2)
        status._hint = None
        self.assertEqual(status._layout_island(), ui.ISLAND_W)
        self.assertTrue(status._hint_layer.isHidden())
        self.assertEqual(status._orb.position().x, ui.ISLAND_W / 2)


class MainThreadDispatchTests(unittest.TestCase):
    """--no-overlay hides the menu bar and pill, never the text delivery."""

    def test_delivery_runs_without_a_ui_and_ui_updates_do_not(self):
        scheduled = []
        ui_call, deliver_call = scrivo.main_thread_dispatch(
            has_ui=False, call_after=lambda method, *args: scheduled.append((method, args)))
        ui_call("refresh_history")
        deliver_call("inject_when_clear", "hello", 0)
        self.assertEqual(scheduled, [("inject_when_clear", ("hello", 0))])

    def test_with_a_ui_both_are_scheduled_in_order(self):
        scheduled = []
        ui_call, deliver_call = scrivo.main_thread_dispatch(
            has_ui=True, call_after=lambda method, *args: scheduled.append(method))
        ui_call("show_error")
        deliver_call("undo_when_clear")
        self.assertEqual(scheduled, ["show_error", "undo_when_clear"])


class CaptureGateTests(unittest.TestCase):
    """Before an update restart, new recordings are refused, then in-flight work drains."""

    def test_closed_gate_refuses_new_starts_and_reopens(self):
        gate = scrivo.CaptureGate()
        with gate.starting() as allowed:
            self.assertTrue(allowed)
        gate.close()
        with gate.starting() as allowed:
            self.assertFalse(allowed)
        gate.reopen()
        with gate.starting() as allowed:
            self.assertTrue(allowed)

    def test_close_waits_for_a_start_already_under_way(self):
        gate = scrivo.CaptureGate()
        inside, release, order = threading.Event(), threading.Event(), []

        def start():
            with gate.starting() as allowed:
                order.append(("start", allowed))
                inside.set()
                release.wait(5)

        starter = threading.Thread(target=start)
        starter.start()
        inside.wait(5)
        closer = threading.Thread(target=lambda: (gate.close(), order.append("closed")))
        closer.start()
        time.sleep(0.05)
        self.assertEqual(order, [("start", True)])  # close() is still waiting
        release.set()
        starter.join(5)
        closer.join(5)
        self.assertEqual(order, [("start", True), "closed"])


class UpdateRestartTests(unittest.TestCase):
    """An update restarts Scrivo only once nothing is recording or transcribing."""

    def test_waits_while_busy_and_returns_once_idle(self):
        busy = iter([True, True, False])
        sleeps = []
        scrivo.wait_until_idle(lambda: next(busy), poll_s=0.5, sleep=sleeps.append)
        self.assertEqual(sleeps, [0.5, 0.5])

    def test_returns_at_once_when_already_idle(self):
        sleeps = []
        self.assertTrue(scrivo.wait_until_idle(lambda: False, sleep=sleeps.append))
        self.assertEqual(sleeps, [])

    def test_quit_waits_at_most_its_deadline(self):
        # Quit finishes the dictation in progress, but never waits forever.
        clock = [0.0]

        def sleep(seconds):
            clock[0] += seconds
        reached = scrivo.wait_until_idle(lambda: True, poll_s=0.5, sleep=sleep,
                                        deadline_s=scrivo.QUIT_DRAIN_S, clock=lambda: clock[0])
        self.assertFalse(reached)
        self.assertEqual(clock[0], scrivo.QUIT_DRAIN_S)
        self.assertEqual(scrivo.QUIT_DRAIN_S, 10.0)


def _closure_source(name: str) -> str:
    """The source of one nested def inside scrivo.run, up to the next statement at its indent."""
    import re
    runtime = inspect.getsource(scrivo.run)
    start = runtime.index(f"def {name}(")
    indent = runtime[:start].rsplit("\n", 1)[1]
    following = re.compile(rf"\n{indent}\S").search(runtime, start)
    return runtime[start:following.start() if following else None]


class QuitDrainTests(unittest.TestCase):
    """Quit waits for the finished dictation to be pasted, not just transcribed."""

    def test_a_paste_scheduled_after_the_queue_drains_still_counts_as_busy(self):
        jobs, deliveries, main_loop, pasted = queue.Queue(), scrivo.PendingDeliveries(), [], []
        _, deliver_call = scrivo.main_thread_dispatch(
            has_ui=True, call_after=lambda method, *args: main_loop.append((method, args)),
            deliveries=deliveries)
        capture = Mock()
        capture.is_active.return_value = False

        def paste(text):
            pasted.append(text)
            deliveries.finish()

        jobs.put("capture")
        jobs.get()
        deliver_call(paste, "hello")  # the worker schedules the paste on the main loop …
        jobs.task_done()              # … and only then marks the job done
        self.assertEqual(jobs.unfinished_tasks, 0)
        self.assertTrue(scrivo.dictation_in_flight(capture, jobs, deliveries))
        method, args = main_loop.pop(0)
        method(*args)
        self.assertEqual(pasted, ["hello"])
        self.assertFalse(scrivo.dictation_in_flight(capture, jobs, deliveries))

    def test_every_delivery_goes_through_the_counted_queue(self):
        # Both deliver_call targets only enqueue; DeliveryQueue finishes each
        # entry (on_done=pending_deliveries.finish) when it delivers, drops or
        # clears it, and posts nothing after shutdown — behaviour pinned in
        # tests/test_delivery.py PendingDeliveriesAccountingTests.
        self.assertIn("delivery.paste(text, in_history)", _closure_source("inject_when_clear"))
        self.assertIn("delivery.undo()", _closure_source("undo_when_clear"))

    def test_update_and_engine_restarts_also_wait_for_the_paste(self):
        # The update restart and the engine switch restart the process too, so
        # they must count a scheduled paste as busy, like Quit: every in-flight
        # check in run() goes through Lifecycle.busy (dictation_in_flight).
        # Quit, Restart and the signal drain are behaviour-tested in
        # tests/test_mac_lifecycle.py.
        self.assertIn("busy=lambda: dictation_in_flight(capture, jobs, pending_deliveries,",
                      inspect.getsource(scrivo.run))
        # The engine switch and Update Now ask Lifecycle.busy, before and while
        # draining: tests/test_app_controller.py EngineTests and ApplyUpdateTests,
        # tests/test_mac_lifecycle.py EngineSwitchTests.

    def test_a_finishing_capture_stays_in_flight_until_its_job_is_queued(self):
        # capture.end() makes the capture inactive before the job is queued; the
        # count must cover that gap or a Quit poll landing in it loses the dictation.
        finish = _closure_source("on_finish")
        self.assertLess(finish.index("pending_deliveries.add()"), finish.index("finish_capture()"))
        self.assertIn("pending_deliveries.finish()", finish.split("finally:", 1)[1])
        body = _closure_source("finish_capture")
        self.assertLess(body.index("capture.end("), body.index('shutdown.enqueue(jobs, ("live"'))

    def test_the_update_restart_wait_has_a_deadline(self):
        # The gate is closed while it waits, so a wedged native call (or a paste
        # count that never drops) must not refuse recordings forever. Update Now
        # passes this deadline to its drain (tests/test_app_controller.py
        # ApplyUpdateTests); [N28] only Scrivo.app offers or runs Update Now
        # (CheckUpdatesTests, ApplyUpdateTests.test_a_terminal_run_never_updates_itself).
        self.assertGreater(scrivo.UPDATE_DRAIN_DEADLINE_S, scrivo.HANDS_FREE_MAX_S)  # outlasts any recording


class FinishCaptureNowTests(unittest.TestCase):
    """Finish now and Quit both end a capture the gesture engine lost track of."""

    def _call(self, *, forced: bool, active: bool):
        engine, capture, on_finish = Mock(), Mock(), Mock()
        engine.force_finish.return_value = forced
        capture.is_active.return_value = active
        return scrivo.end_capture_now(engine, capture, on_finish), on_finish

    def test_gesture_recording_is_finished_by_the_engine(self):
        ended, on_finish = self._call(forced=True, active=True)
        self.assertEqual(ended, "gesture")
        on_finish.assert_not_called()

    def test_orphan_capture_is_transcribed_not_dropped(self):
        ended, on_finish = self._call(forced=False, active=True)
        self.assertEqual(ended, "orphan")
        on_finish.assert_called_once_with()

    def test_nothing_recording_changes_nothing(self):
        ended, on_finish = self._call(forced=False, active=False)
        self.assertIsNone(ended)
        on_finish.assert_not_called()

    def test_quit_and_finish_now_share_it(self):
        # Quit, Restart and the signal drain end it through Lifecycle.end_recording;
        # the menu's Finish now through AppController.finish_now, with the
        # gesture's own on_finish (tests/test_app_controller.py FinishNowTests;
        # through the real run(), test_nemotron_pipeline's dictation ends with it).
        self.assertIn("end_recording=lambda: end_capture_now(engine, capture, on_finish)",
                      inspect.getsource(scrivo.run))


class EngineInstallerTests(unittest.TestCase):
    """An engine download that fails to start or hangs reports it; it never raises."""

    def test_success_failure_launch_error_and_timeout(self):
        def finished(code, out="", err=""):
            return lambda argv, **kwargs: subprocess.CompletedProcess(argv, code, out, err)

        self.assertEqual(scrivo.run_installer(["x"], run=finished(0, "done\n")), (True, "done"))
        self.assertEqual(scrivo.run_installer(["x"], run=finished(1, "", "boom\n")), (False, "boom"))

        def cannot_start(argv, **kwargs):
            raise OSError(35, "Resource temporarily unavailable")
        ok, detail = scrivo.run_installer(["x"], run=cannot_start)
        self.assertFalse(ok)
        self.assertIn("could not start", detail)

        def hangs(argv, **kwargs):
            self.assertEqual(kwargs["timeout"], scrivo.INSTALL_TIMEOUT_S)
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        ok, detail = scrivo.run_installer(["x"], run=hangs)
        self.assertFalse(ok)
        self.assertIn("timed out", detail)


@unittest.skipUnless(sys.platform == 'darwin', 'AppKit menu bar')
class TapWarningTests(unittest.TestCase):
    """A deaf hotkey's ⚠ stays in the menu bar until the tap recovers."""

    def test_hiding_the_pill_keeps_the_warning(self):
        import ui
        titles = []
        status = ui.StatusUI.__new__(ui.StatusUI)
        status._mode, status._hint, status._live_text = "idle", None, ""
        status._indicator_state, status._visibility_generation, status._level_timer = "idle", 0, None
        status._status = types.SimpleNamespace(button=lambda: types.SimpleNamespace(setTitle_=titles.append))
        status._orb = Mock()
        status._panel = Mock()
        status.set_tap_health(False)
        status.hide()
        self.assertEqual(titles[-1], "⚠")
        status.set_tap_health(True)
        status.hide()
        self.assertEqual(titles[-1], "◦")


class AppLogRotationTests(unittest.TestCase):
    """The app's log is rotated while Scrivo runs, not only at launch."""

    def test_a_log_that_grows_past_its_limit_is_rotated_and_reopened(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scrivo.log"
            path.write_text("x" * 50)
            reopened = []
            with patch.object(scrivo_log, "_point_output_at", reopened.append), \
                    patch.dict(scrivo_log._app_log, {"path": path, "max_bytes": 100}):
                scrivo.keep_app_log_small()
                self.assertEqual(reopened, [])           # under the limit: untouched
                path.write_text("x" * 150)
                scrivo.keep_app_log_small()
            self.assertEqual(reopened, [path])
            self.assertEqual(path.with_suffix(".log.1").read_text(), "x" * 150)
            self.assertFalse(path.exists())             # the reopen creates the fresh file

    def test_routing_rotates_a_large_log_points_output_at_it_and_records_it(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            pointed = []
            with patch.object(scrivo_log, "_point_output_at", pointed.append), \
                    patch.dict(scrivo_log._app_log, {"path": None, "max_bytes": 0}):
                path = scrivo.route_app_logs(data, max_bytes=100)
                self.assertEqual(dict(scrivo_log._app_log), {"path": path, "max_bytes": 100})
                self.assertEqual(path, data / "logs" / "scrivo.log")
                if os.name == "posix":                         # Windows has no owner-only mode bits
                    self.assertEqual(os.stat(data / "logs").st_mode & 0o777, 0o700)
                path.write_text("x" * 150)                    # an oversized log from the last run
                scrivo.route_app_logs(data, max_bytes=100)
            self.assertEqual(pointed, [path, path])
            self.assertEqual(path.with_suffix(".log.1").read_text(), "x" * 150)

    def test_nothing_happens_when_logs_are_not_routed(self):
        with patch.object(scrivo_log, "_point_output_at") as reopen, \
                patch.dict(scrivo_log._app_log, {"path": None, "max_bytes": 100}):
            scrivo.keep_app_log_small()
        reopen.assert_not_called()

    def test_n35_two_threads_rotating_at_once_neither_raises(self):
        real_replace, errors, reopened, callers = os.replace, [], [], []

        def slow_replace(source, target):
            # Both threads saw the oversized log before either moved it; the
            # second mover is slower, so it finds the first one's fresh log.
            callers.append(1)
            time.sleep(0.05 * len(callers) ** 2)
            real_replace(source, target)

        def reopen(path):
            reopened.append(path)
            path.touch()  # what _point_output_at's O_CREAT does

        def write_a_line():
            try:
                scrivo.log("a line")
            except Exception as exc:  # what ended the transcription worker
                errors.append(exc)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scrivo.log"
            path.write_text("x" * 150)
            with patch.object(scrivo_log, "_point_output_at", reopen), \
                    patch.dict(scrivo_log._app_log, {"path": path, "max_bytes": 100}), \
                    patch.object(scrivo.os, "replace", slow_replace), \
                    patch("sys.stderr", new=open(os.devnull, "w")):
                threads = [threading.Thread(target=write_a_line) for _ in range(2)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(5)
                sys.stderr.close()
            rotated = path.with_suffix(".log.1").read_text()
        self.assertEqual(errors, [])
        self.assertEqual(reopened, [path])                    # rotated once, not twice
        self.assertEqual(rotated, "x" * 150)                  # the old log was not overwritten

    def test_n35_a_failing_log_write_never_raises_into_the_caller(self):
        def disk_full(*_args, **_kwargs):
            raise OSError(28, "No space left on device")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scrivo.log"
            path.write_text("x" * 150)
            with patch.object(scrivo_log, "_point_output_at", disk_full), \
                    patch.dict(scrivo_log._app_log, {"path": path, "max_bytes": 100}), \
                    patch("sys.stderr", new=open(os.devnull, "w")):
                scrivo.log("a line")                           # must not raise
                with patch("builtins.print", disk_full):
                    scrivo.log("another line")
                sys.stderr.close()


class TelemetryTests(unittest.TestCase):
    def test_n37_importing_scrivo_turns_hub_telemetry_off(self):
        environment = {key: value for key, value in os.environ.items()
                       if key not in {"HF_HUB_DISABLE_TELEMETRY", "DISABLE_TELEMETRY"}}
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as data:
            environment["SCRIVO_DATA_DIR"] = data
            result = subprocess.run(
                [sys.executable, "-c", "import scrivo, huggingface_hub.constants as c; "
                                       "print(c.HF_HUB_DISABLE_TELEMETRY)"],
                cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.stdout.strip().splitlines()[-1:], ["True"], result.stderr[-800:])


class SetupCommandTests(unittest.TestCase):
    def test_n42_setup_for_parakeet_also_installs_voice_detection(self):
        import vad
        installed = []
        fake_parakeet = types.SimpleNamespace(path="/models/parakeet-snapshot")
        with patch.object(sys, "argv", ["scrivo.py", "setup", "--profile", "parakeet"]), \
                patch.object(scrivo, "LocalParakeet", lambda repo: fake_parakeet), \
                patch.object(scrivo, "LocalWhisper", side_effect=AssertionError("Whisper not asked for")), \
                patch.object(vad, "install", lambda allow_download: installed.append(allow_download)
                             or Path("/data/models/silero_vad.onnx")), \
                patch("builtins.print"):
            scrivo.main()
        self.assertEqual(len(installed), 1)


class SaveAudioNameTests(unittest.TestCase):
    """Save audio to Desktop never overwrites an earlier save."""

    def test_names_carry_seconds_and_never_reuse_a_file(self):
        import datetime
        with tempfile.TemporaryDirectory() as folder:
            desktop = Path(folder)
            moment = datetime.datetime(2026, 10, 9, 1, 53, 7).timestamp()
            first = scrivo.desktop_audio_path(desktop, moment)
            self.assertEqual(first.name, "scrivo-20261009-015307.wav")
            first.write_bytes(b"one")
            second = scrivo.desktop_audio_path(desktop, moment)
            self.assertEqual(second.name, "scrivo-20261009-015307-2.wav")
            second.write_bytes(b"two")
            self.assertEqual(scrivo.desktop_audio_path(desktop, moment + 20).name, "scrivo-20261009-015327.wav")
            self.assertEqual(scrivo.desktop_audio_path(desktop, moment).name, "scrivo-20261009-015307-3.wav")

    def test_a_name_is_reserved_the_moment_it_is_picked(self):
        # Two saves that both pick a name before either copies must not share it.
        with tempfile.TemporaryDirectory() as folder:
            desktop = Path(folder)
            first = scrivo.desktop_audio_path(desktop, 1_760_000_000.0)
            second = scrivo.desktop_audio_path(desktop, 1_760_000_000.0)
            self.assertNotEqual(first, second)
            self.assertTrue(first.exists() and second.exists())

    def test_concurrent_saves_of_one_entry_make_two_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, desktop = root / "entry.wav", root / "Desktop"
            source.write_bytes(b"RIFF-audio")
            desktop.mkdir()
            both_ready, saved = threading.Barrier(2), []

            def save():
                both_ready.wait(5)
                saved.append(scrivo.save_audio_copy(source, desktop, 1_760_000_000.0))

            workers = [threading.Thread(target=save) for _ in range(2)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(5)
            self.assertEqual(len(set(saved)), 2)
            self.assertEqual(sorted(path.read_bytes() for path in desktop.iterdir()),
                             [b"RIFF-audio", b"RIFF-audio"])

    def test_missing_audio_leaves_nothing_on_the_desktop(self):
        with tempfile.TemporaryDirectory() as folder:
            desktop = Path(folder)
            with self.assertRaises(FileNotFoundError):
                scrivo.save_audio_copy(desktop / "gone.wav", desktop, 1_760_000_000.0)
            self.assertEqual(list(desktop.iterdir()), [])


class LiveWordsTests(unittest.TestCase):
    """Streaming engines show words while you talk; only finished text is pasted."""

    def test_stream_reports_finals_and_interim_but_nothing_once_closed(self):
        from streaming_audio import StreamingCapture
        capture = StreamingCapture(runtime=None)
        self.assertEqual(capture.live_text(), "")
        capture.stream = types.SimpleNamespace(finals=["Hello there."], partial="this is live")
        self.assertEqual(capture.live_text(), "Hello there. this is live")
        capture.closed = True
        self.assertEqual(capture.live_text(), "")

    def test_live_words_use_the_personal_dictionary(self):
        import dictionary
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dictionary.txt"
            path.write_text("soto => Scrivo\n", encoding="utf-8")
            with patch.object(dictionary, "DICTIONARY_PATH", path):
                self.assertEqual(scrivo.live_preview("Soto turns what I say"), "Scrivo turns what I say")
                self.assertEqual(scrivo.live_preview(""), "")
            with patch.object(dictionary, "load", side_effect=OSError("unreadable")):
                self.assertEqual(scrivo.live_preview("soto as heard"), "soto as heard")

    @unittest.skipUnless(sys.platform == 'darwin', 'AppKit overlay')
    def test_long_text_keeps_its_newest_end(self):
        import ui
        self.assertEqual(ui.live_snippet("short  words"), "short words")
        long = " ".join(f"word{i}" for i in range(30))
        snippet = ui.live_snippet(long)
        self.assertTrue(snippet.startswith("…") and snippet.endswith("word29"))
        self.assertLessEqual(len(snippet), ui.LIVE_CHARS)

    @unittest.skipUnless(sys.platform == 'darwin', 'AppKit overlay')
    def test_live_words_win_over_the_hint_and_refresh_only_while_recording(self):
        import ui
        status = ui.StatusUI.__new__(ui.StatusUI)
        status._panel = status._make_island()
        status._orb = status._make_orb()
        status._hint, status._live_text, status._live_checked = 'tap right ⌘ to stop', '', 0.0
        status._indicator_state = 'recording'
        words = iter(['Hello', 'Hello world'])
        status.live_text_source = lambda: next(words)
        with patch.object(status, '_position') as position:
            status.refresh_live_text()
            status.refresh_live_text()                  # throttled: no second read yet
            status._live_checked = 0.0
            status.refresh_live_text()
        self.assertEqual(position.call_count, 2)
        self.assertEqual(status._live_text, 'Hello world')
        status._layout_island()
        self.assertEqual(status._hint_layer.string(), 'Hello world')
        status._indicator_state = 'transcribing'
        status.live_text_source = lambda: self.fail('not read outside recording')
        status.refresh_live_text()


class CaptureShutdownTests(unittest.TestCase):
    def test_shutdown_starts_one_teardown_and_restart_wait_is_bounded(self):
        service = scrivo.CaptureService()
        gate = threading.Event()
        calls = []

        def blocked_release(**_):
            calls.append(1)
            gate.wait(5)
            return True

        service._release_engine = blocked_release
        service.shutdown()
        service.shutdown()
        started = time.monotonic()
        self.assertFalse(service.wait_released(0.2))
        self.assertLess(time.monotonic() - started, 2.0)
        gate.set()
        self.assertTrue(service.wait_released(2.0))
        self.assertEqual(len(calls), 1)

    def test_wait_without_shutdown_returns_immediately(self):
        self.assertTrue(scrivo.CaptureService().wait_released(0.0))


class StorageAndPermissionTests(unittest.TestCase):
    def test_alpha_override_covers_preferences_history_vad_and_native_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = dict(os.environ, SCRIVO_DATA_DIR=str(root / 'state'), SCRIVO_HF_HOME=str(root / 'cache'))
            code = ('import json, history, speech_config, vad, nemotron_backend, scrivo; '
                    'print(json.dumps([str(history.STORE_DIR), '
                    'str(speech_config.DEFAULT_LANGUAGE_MODE_PATH), '
                    'str(speech_config.DEFAULT_ENGINE_MODE_PATH), str(vad.MODEL_PATH), '
                    'str(nemotron_backend.INSTALL_ROOT), str(scrivo.SCRIVO_HF_HOME)]))')
            result = subprocess.run([sys.executable, '-c', code], env=env, check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout), [str(root/'state'), str(root/'state/language-mode'),
                str(root/'state/engine-mode'), str(root/'state/models/silero_vad.onnx'),
                str(root/'state/nemotron'), str(root/'cache')])
            self.assertFalse((root/'state').exists(), 'import must not create/migrate state')

    @unittest.skipUnless(sys.platform == 'darwin', 'macOS default paths; Windows defaults: tests/test_win_alpha.py')
    def test_blank_overrides_fall_back_to_defaults_never_the_working_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            env = dict(os.environ, SCRIVO_DATA_DIR='  ', SCRIVO_HF_HOME='', PYTHONPATH=str(ROOT))
            code = 'import json, scrivo_paths as p; print(json.dumps([str(p.DATA_DIR), str(p.MODEL_CACHE_DIR)]))'
            result = subprocess.run([sys.executable, '-c', code], env=env, cwd=temporary,
                                    check=True, capture_output=True, text=True)
            data, cache = json.loads(result.stdout)
            home = str(Path.home())
            self.assertEqual(data, home + '/Library/Application Support/sotto')
            self.assertEqual(cache, home + '/Library/Application Support/Sotto/huggingface')
            self.assertNotIn(os.path.realpath(temporary), (os.path.realpath(data), os.path.realpath(cache)))

    @unittest.skipUnless(sys.platform == 'darwin', 'AVFoundation microphone permission is macOS-only')
    def test_denied_microphone_does_not_request_it_again(self):
        with patch('AVFoundation.AVCaptureDevice') as device:
            device.authorizationStatusForMediaType_.return_value = 2
            self.assertFalse(scrivo.microphone_permission(request=True))
            device.requestAccessForMediaType_completionHandler_.assert_not_called()

    @unittest.skipUnless(sys.platform == 'darwin', 'AVFoundation microphone permission is macOS-only')
    def test_undecided_microphone_uses_the_permission_callback(self):
        with patch('AVFoundation.AVCaptureDevice') as device:
            device.authorizationStatusForMediaType_.return_value = 0
            device.requestAccessForMediaType_completionHandler_.side_effect = lambda media, callback: callback(True)
            self.assertTrue(scrivo.microphone_permission(request=True))

    @unittest.skipUnless(sys.platform == 'darwin', 'macOS doctor (Accessibility + CoreAudio probe)')
    def test_doctor_fails_when_input_probe_fails_even_with_permissions(self):
        with patch('ApplicationServices.AXIsProcessTrusted', return_value=True), \
             patch.object(scrivo, 'microphone_permission', return_value=True), \
             patch('subprocess.run', return_value=subprocess.CompletedProcess([], 1, stdout='', stderr='')):
            with self.assertRaises(SystemExit) as error:
                scrivo.doctor()
            self.assertEqual(error.exception.code, 1)


class ModelAndTemplateTests(unittest.TestCase):
    DEFAULT = 'mlx-community/whisper-large-v3-turbo'

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.snapshot = Path(temporary.name) / 'snapshot'
        self.snapshot.mkdir()
        (self.snapshot / 'config.json').write_text('{}')
        (self.snapshot / 'weights.safetensors').write_bytes(b'weights')
        self.partial = Path(temporary.name) / 'partial'
        self.partial.mkdir()
        (self.partial / 'config.json').write_text('{}')

    def resolve(self, repo, *, offline, results):
        backend = Mock()
        with patch('offline_runtime.offline_requested', return_value=offline), \
             patch('huggingface_hub.snapshot_download', side_effect=results) as snapshot:
            try:
                return scrivo.LocalWhisper(backend, repo), backend, snapshot
            except scrivo.ModelUnavailable as exc:
                return exc, backend, snapshot

    def test_every_builtin_whisper_profile_is_pinned_to_an_exact_commit(self):
        for profile in MODEL_PROFILES.values():
            if profile.backend == 'whisper':
                self.assertRegex(MODEL_REVISIONS.get(profile.repo, ''), r'^[0-9a-f]{40}$', profile.repo)

    def test_cached_pinned_snapshot_is_used_without_any_hub_request(self):
        local, backend, snapshot = self.resolve(self.DEFAULT, offline=False, results=[str(self.snapshot)])
        self.assertEqual(snapshot.call_count, 1)
        self.assertTrue(snapshot.call_args.kwargs['local_files_only'])
        self.assertEqual(snapshot.call_args.kwargs['revision'], MODEL_REVISIONS[self.DEFAULT])
        local.transcribe([0.1], path_or_hf_repo=self.DEFAULT, language='en')
        self.assertEqual(backend.transcribe.call_args.kwargs['path_or_hf_repo'], str(self.snapshot))

    def test_offline_missing_model_is_a_plain_error_and_never_downloads(self):
        error, _backend, snapshot = self.resolve(self.DEFAULT, offline=True,
                                                 results=FileNotFoundError('missing'))
        self.assertIsInstance(error, scrivo.ModelUnavailable)
        self.assertIn('not downloaded yet', str(error))
        self.assertEqual(snapshot.call_count, 1)
        self.assertTrue(snapshot.call_args.kwargs['local_files_only'])

    def test_first_online_start_downloads_the_pinned_revision_once(self):
        local, backend, snapshot = self.resolve(
            self.DEFAULT, offline=False, results=[FileNotFoundError('missing'), str(self.snapshot)])
        self.assertEqual(snapshot.call_count, 2)
        download = snapshot.call_args_list[1].kwargs
        self.assertNotIn('local_files_only', download)
        self.assertEqual(download['revision'], MODEL_REVISIONS[self.DEFAULT])
        for _ in range(2):
            local.transcribe([0.1], path_or_hf_repo=self.DEFAULT)
        self.assertEqual(snapshot.call_count, 2)
        self.assertEqual(backend.transcribe.call_args.kwargs['path_or_hf_repo'], str(self.snapshot))

    def test_incomplete_cached_snapshot_is_completed_online(self):
        local, _backend, snapshot = self.resolve(
            self.DEFAULT, offline=False, results=[str(self.partial), str(self.snapshot)])
        self.assertEqual(snapshot.call_count, 2)
        self.assertEqual(local.path, str(self.snapshot))

    def test_download_failure_is_a_plain_error(self):
        error, _backend, _snapshot = self.resolve(
            self.DEFAULT, offline=False, results=[FileNotFoundError('missing'), OSError('network down')])
        self.assertIsInstance(error, scrivo.ModelUnavailable)
        self.assertIn('could not download', str(error))

    def test_custom_model_has_no_pin_but_still_resolves_cache_first(self):
        _local, _backend, snapshot = self.resolve('fixture/model', offline=False, results=[str(self.snapshot)])
        self.assertIsNone(snapshot.call_args.kwargs['revision'])
        self.assertTrue(snapshot.call_args.kwargs['local_files_only'])

    @unittest.skipUnless(sys.platform == 'darwin', 'LaunchAgent templates use POSIX paths')
    def test_every_test_module_source_and_workflow_ships_in_the_archive(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('build_alpha_source', ROOT / 'scripts/build_alpha_source.py')
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        shipped = set(builder.PUBLIC_FILES)
        missing = [path.relative_to(ROOT).as_posix()
                   for path in [*ROOT.glob('tests/*.py'), *ROOT.glob('*.py'),
                                *ROOT.glob('.github/workflows/*.yml')]
                   if path.relative_to(ROOT).as_posix() not in shipped]
        self.assertEqual(missing, [])

    def test_automatic_chooses_only_among_the_allowed_languages(self):
        local, backend, _snapshot = self.resolve(self.DEFAULT, offline=False, results=[str(self.snapshot)])
        with patch.object(scrivo.LocalWhisper, 'one_pass', return_value=('pt', None)) as detect:
            local.allowed_languages = lambda: ('en', 'pt')
            local.transcribe([0.1])
            self.assertEqual(backend.transcribe.call_args.kwargs['language'], 'pt')
            self.assertEqual(local.last_language, 'pt')
            local.transcribe([0.1], language='en')           # an explicit choice is never overridden
            self.assertEqual(backend.transcribe.call_args.kwargs['language'], 'en')
            self.assertIsNone(local.last_language)
            local.allowed_languages = lambda: ('fr',)        # one language: no detection needed
            local.transcribe([0.1])
            self.assertEqual(backend.transcribe.call_args.kwargs['language'], 'fr')
            local.allowed_languages = None                   # no set: Whisper's own detection
            local.transcribe([0.1])
            self.assertNotIn('language', backend.transcribe.call_args.kwargs)
        self.assertEqual(detect.call_count, 1)  # only the multi-language Automatic call

    def test_short_automatic_clips_use_one_pass_and_fall_back_when_unsure(self):
        local, backend, _snapshot = self.resolve(self.DEFAULT, offline=False, results=[str(self.snapshot)])
        local.allowed_languages = lambda: ('en', 'pt')
        short, long = [0.0] * 16000, [0.0] * (31 * 16000)
        with patch.object(scrivo.LocalWhisper, 'one_pass', return_value=('pt', {'text': 'olá'})) as one_pass:
            self.assertEqual(local.transcribe(short)['text'], 'olá')
            backend.transcribe.assert_not_called()
        with patch.object(scrivo.LocalWhisper, 'one_pass', return_value=('en', None)):
            local.transcribe(short)                            # unsure decode: the full pipeline decides
            self.assertEqual(backend.transcribe.call_args.kwargs['language'], 'en')
        with patch.object(scrivo.LocalWhisper, 'detect_language', return_value='pt') as detect:
            local.transcribe(long)                             # beyond one window: detect then transcribe
            self.assertEqual(detect.call_count, 1)
            local.transcribe(short, initial_prompt='Scrivo')    # a glossary prompt needs the full path
            self.assertEqual(backend.transcribe.call_args.kwargs['language'], 'pt')
        self.assertEqual(one_pass.call_count, 1)

    def test_fast_speed_uses_the_trimmed_one_pass_even_for_a_fixed_language(self):
        local, backend, _snapshot = self.resolve(self.DEFAULT, offline=False, results=[str(self.snapshot)])
        local.speed = lambda: 'fast'
        short = [0.0] * 16000
        with patch.object(scrivo.LocalWhisper, 'one_pass', return_value=('en', {'text': 'hi'})) as one_pass:
            self.assertEqual(local.transcribe(short, language='en')['text'], 'hi')
            self.assertEqual(one_pass.call_args.kwargs, {'language': 'en', 'trim': True})
            self.assertIsNone(local.last_language)      # an explicit language is not "detected"
        with patch.object(scrivo.LocalWhisper, 'one_pass', return_value=('en', None)):
            local.transcribe(short, language='en')      # unsure: full accurate pipeline
            self.assertEqual(backend.transcribe.call_args.kwargs['language'], 'en')
        local.speed = lambda: 'accurate'
        backend.reset_mock()
        with patch.object(scrivo.LocalWhisper, 'one_pass') as one_pass:
            local.transcribe(short, language='en')      # accurate + fixed: the library path
            one_pass.assert_not_called()
        backend.transcribe.assert_called_once()

    @unittest.skipUnless(sys.platform == 'darwin', 'patches mlx_whisper decoding (macOS-only)')
    def test_one_pass_never_returns_a_decode_that_ran_out_of_tokens(self):
        local, _backend, _snapshot = self.resolve(self.DEFAULT, offline=False, results=[str(self.snapshot)])

        def decode_with(tokens):
            result = types.SimpleNamespace(tokens=tokens, no_speech_prob=0.0, avg_logprob=-0.1,
                                           compression_ratio=1.2, text="partial")
            task = Mock(sample_len=224)
            task.run.return_value = [result]
            return lambda *_a, **_k: task

        def fake_decoding(tokens):  # never import the real mlx_whisper into the test process
            module = types.SimpleNamespace(DecodingOptions=lambda **options: options,
                                           DecodingTask=decode_with(tokens))
            return patch.dict(sys.modules, {'mlx_whisper': types.SimpleNamespace(decoding=module),
                                            'mlx_whisper.decoding': module})

        with patch.object(scrivo.LocalWhisper, '_model', return_value=Mock()), \
             patch.object(scrivo.LocalWhisper, '_features', return_value=Mock()):
            with fake_decoding([1] * 224):
                self.assertEqual(local.one_pass([0.0], (), language='pt'), ('pt', None))
            with fake_decoding([1] * 20):
                self.assertEqual(local.one_pass([0.0], (), language='pt')[1]['text'], 'partial')


class ChooseLanguageTests(unittest.TestCase):
    def test_prefers_the_allowed_set_but_never_translates_clearly_other_speech(self):
        choose = scrivo.choose_language
        self.assertEqual(choose({"en": 0.99, "pt": 0.005, "es": 0.005}, ("en", "pt")), "en")
        self.assertEqual(choose({"cy": 0.45, "en": 0.40, "pt": 0.01}, ("en", "pt")), "en")   # ambiguous: stay
        self.assertEqual(choose({"es": 0.86, "en": 0.11, "pt": 0.001}, ("en", "pt")), "es")  # clearly Spanish
        self.assertEqual(choose({"es": 0.6, "en": 0.3}, ("en", "pt")), "en")                 # in-set not weak
        self.assertEqual(choose({"fr": 0.7, "en": 0.2}, ()), "fr")                           # no set: plain argmax
