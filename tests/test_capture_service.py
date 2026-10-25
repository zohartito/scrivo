"""CaptureService and its runtime wiring, with every CoreAudio / AVFoundation
entry point replaced by an in-process fake: no microphone is opened, no
event tap installed, no event posted.

Findings covered (audit ledger ids): F12 teardown releases every resource,
F17 per-block sample rates after a mid-recording route change, F21 idle
release honours -1 and never races a press, F41 doctor probes the pinned
input bus, F45 a failed microphone start surfaces, F53 the CFString from
name_of is released.
"""

import ast
import contextlib
import ctypes
import os
import queue
import sys
import threading
import time
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np

import capture as capture_module
import scrivo


# -- fake AVFoundation ---------------------------------------------------------

class FakePtr:
    def __init__(self, samples):
        self._bytes = np.asarray(samples, dtype=np.float32).tobytes()

    def as_buffer(self, n):
        return self._bytes[: 4 * n]


class FakeFormat:
    def __init__(self, rate, channels=1):
        self._rate = rate
        self._channels = channels

    def sampleRate(self):
        return self._rate

    def channelCount(self):
        return self._channels


class FakeBuffer:
    def __init__(self, samples, rate):
        self._samples = np.asarray(samples, dtype=np.float32)
        self._rate = rate

    def frameLength(self):
        return len(self._samples)

    def floatChannelData(self):
        return [FakePtr(self._samples)]

    def format(self):
        return FakeFormat(self._rate)


class FakeWhen:
    def __init__(self, sample_time):
        self._t = sample_time

    def sampleTime(self):
        return self._t


class FakeNode:
    def __init__(self, rate=48000.0, remove_raises=False):
        self.rate = rate
        self.remove_raises = remove_raises
        self.tap = None
        self.calls = []

    def removeTapOnBus_(self, bus):
        self.calls.append("removeTap")
        if self.remove_raises:
            raise RuntimeError("AVAudioNode removeTapOnBus: device gone")
        self.tap = None

    def inputFormatForBus_(self, bus):
        self.calls.append("inputFormat")
        return FakeFormat(self.rate)

    def outputFormatForBus_(self, bus):
        self.calls.append("outputFormat")
        return FakeFormat(96000.0, 2)   # the system default, not the pinned mic

    def installTapOnBus_bufferSize_format_block_(self, bus, size, fmt, block):
        self.calls.append(f"installTap@{fmt.sampleRate():.0f}")
        self.tap = block


class FakeEngine:
    def __init__(self, node=None, start_ok=True, stop_raises=False):
        self.node = node or FakeNode()
        self.start_ok = start_ok
        self.stop_raises = stop_raises
        self.running = False
        self.calls = []

    def inputNode(self):
        return self.node

    def prepare(self):
        self.calls.append("prepare")

    def startAndReturnError_(self, _):
        self.calls.append("start")
        if self.start_ok:
            self.running = True
            return True, None
        return False, "Error Domain=com.apple.coreaudio.avfaudio Code=-10868"

    def stop(self):
        self.calls.append("stop")
        if self.stop_raises:
            raise RuntimeError("AVAudioEngine stop: HAL wedged")
        self.running = False

    def isRunning(self):
        return self.running


def fake_avfoundation(engine_factory):
    """capture imports AVAudioEngine lazily inside _start_engine_locked."""
    module = types.ModuleType("AVFoundation")

    class AVAudioEngine:
        @classmethod
        def alloc(cls):
            return cls

        @classmethod
        def init(cls):
            return engine_factory()

    module.AVAudioEngine = AVAudioEngine
    module.AVAudioEngineConfigurationChangeNotification = "config-change"
    return mock.patch.dict(sys.modules, {"AVFoundation": module})


def tone(freq, seconds, rate):
    t = np.arange(int(seconds * rate)) / rate
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def feed(capture, samples, rate, start_sample=0, block=4096):
    """Drive the real CaptureService._tap with fake AVAudioPCMBuffers."""
    for offset in range(0, len(samples), block):
        chunk = samples[offset: offset + block]
        capture._tap(FakeBuffer(chunk, rate), FakeWhen(start_sample + offset))


def fourcc(code: str) -> int:
    return int.from_bytes(code.encode("ascii"), "big")


# -- F53: the CFString handed to name_of is a +1 reference -------------------

class FakeCoreAudio:
    """One built-in input device whose kAudioObjectPropertyName ('lnam') is a
    CFStringRef the caller owns, exactly as CoreAudio hands it out."""
    NAME_REF = 0xC0FFEE

    def AudioObjectGetPropertyData(self, obj_id, addr_ref, _qsize, _qdata, size_ref, buf_ref):
        selector, scope = addr_ref._obj.selector, addr_ref._obj.scope
        size, buf = size_ref._obj, buf_ref._obj
        if selector == fourcc("dev#"):
            buf[0], size.value = 42, 4
        elif selector == fourcc("tran"):
            buf.value = fourcc("bltn")
        elif selector == fourcc("stm#"):
            if scope == fourcc("inpt"):
                buf[0], size.value = 7, 4
            else:
                size.value = 0
        elif selector == fourcc("term"):
            buf.value = 0x201
        elif selector == fourcc("lnam"):
            buf.value = self.NAME_REF
        return 0


class FakeCoreFoundation:
    def __init__(self):
        self.released = []

    def CFStringGetCString(self, ref, buf, size, encoding):
        buf.value = b"MacBook Pro Microphone"
        return 1

    def CFRelease(self, ref):
        self.released.append(ref.value if isinstance(ref, ctypes.c_void_p) else ref)


class DeviceNameLeakTest(unittest.TestCase):
    def test_every_device_name_cfstring_is_released_after_reading(self):
        core_foundation = FakeCoreFoundation()

        def cdll(path, *args, **kwargs):
            if path.endswith("/CoreAudio"):
                return FakeCoreAudio()
            if path.endswith("/CoreFoundation"):
                return core_foundation
            raise AssertionError(f"unexpected library {path}")

        with mock.patch("ctypes.CDLL", cdll):
            devices = capture_module._audio_devices()
        self.assertEqual([d["name"] for d in devices], ["MacBook Pro Microphone"])
        self.assertEqual(core_foundation.released, [FakeCoreAudio.NAME_REF],
                         "kAudioObjectPropertyName is a +1 CFStringRef; name_of must CFRelease it")


# -- F12: teardown releases every resource even if one step raises -----------

class TeardownTest(unittest.TestCase):
    def setUp(self):
        self.logs = []
        patcher = mock.patch.object(capture_module, "log", self.logs.append)
        patcher.start()
        self.addCleanup(patcher.stop)

    def live_capture(self, engine):
        capture = scrivo.CaptureService()
        engine.running = True
        capture._engine_obj = engine
        capture._engine, capture._node = engine, engine.node
        capture._last_device_scan = scrivo.time.monotonic()  # skip the device scan
        return capture

    def test_engine_is_stopped_even_if_tap_removal_raises(self):
        engine = FakeEngine(FakeNode(remove_raises=True))
        capture = self.live_capture(engine)
        self.assertTrue(capture._release_engine())           # key-up release
        self.assertIn("stop", engine.calls)
        self.assertFalse(engine.isRunning(), "idle Scrivo left the mic engine running")
        for _ in range(3):
            capture.tick()                                   # 1 Hz idle health check
        self.assertFalse(engine.isRunning())

    def test_release_reports_failure_and_keeps_the_handle_until_the_engine_stops(self):
        engine = FakeEngine(stop_raises=True)
        capture = self.live_capture(engine)
        self.assertFalse(capture._release_engine(), "release claimed success while the engine ran")
        self.assertIs(capture._engine, engine)               # tick retries, nothing is orphaned
        self.assertEqual(engine.node.calls, ["removeTap"])
        engine.stop_raises = False
        capture.tick()                                       # idle release retried
        self.assertFalse(engine.isRunning())
        self.assertIsNone(capture._engine)
        self.assertIn("○ mic released (idle) — wakes on next press", self.logs)


# -- F21: tick's idle release honours -1 and never races a press ---------------

class IdleReleaseTest(unittest.TestCase):
    def setUp(self):
        self.logs = []
        patcher = mock.patch.object(capture_module, "log", self.logs.append)
        patcher.start()
        self.addCleanup(patcher.stop)

    def live_capture(self, idle_release):
        capture = scrivo.CaptureService()
        engine = FakeEngine()
        engine.running = True
        capture._engine_obj = capture._engine = engine
        capture._node = engine.node
        capture.idle_release_s = idle_release
        capture._last_device_scan = scrivo.time.monotonic()  # skip the device scan
        return capture, engine

    def test_negative_idle_release_means_never_for_tick_too(self):
        capture, engine = self.live_capture(-1)
        capture.release_soon()                               # already honours -1
        capture._last_use -= 3600                            # idle for an hour
        for _ in range(3):
            capture.tick()
        self.assertTrue(engine.isRunning(), "--idle-release -1 engine released by tick()")
        self.assertIs(capture._engine, engine)

    def test_a_press_during_ticks_release_keeps_its_engine_and_audio(self):
        capture, engine = self.live_capture(30.0)            # retained engine, ring warm
        feed(capture, tone(200, 2.0, 48000), 48000)
        capture._last_use -= 31                              # idle past the threshold
        real_release = capture._release_engine

        def press_lands_first(**kwargs):                     # between tick's read and its release
            capture.begin()
            return real_release(**kwargs)

        with mock.patch.object(capture, "_release_engine", press_lands_first):
            capture.tick()
        self.assertTrue(capture.is_active())
        self.assertIs(capture._engine, engine, "tick released the engine under a new capture")
        self.assertTrue(engine.isRunning())
        feed(capture, tone(300, 3.0, 48000), 48000, start_sample=96000)
        raw = capture.end()
        self.assertGreaterEqual(len(raw) / 48000, 3.0)       # pre-roll + the 3 s spoken


# -- F17: per-block sample rates survive a mid-recording route change ----------

class InlineThread:
    """threading.Thread stand-in that runs the target on start(): the route
    rebuild happens right where the test can see it."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        self._target, self._args, self._kwargs = target, args, kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)


def dominant_hz(samples, rate):
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    return np.fft.rfftfreq(len(samples), 1 / rate)[int(np.argmax(spectrum))]


class JoinCaptureBlocksTest(unittest.TestCase):
    def test_same_rate_blocks_concatenate_untouched(self):
        blocks = [tone(440, 0.1, 48000), tone(440, 0.1, 48000)]
        joined = scrivo.join_capture_blocks(blocks, [48000.0, 48000.0], 48000.0)
        np.testing.assert_array_equal(joined, np.concatenate(blocks))

    def test_each_same_rate_run_is_resampled_before_joining(self):
        blocks = [tone(440, 0.5, 48000), tone(440, 0.5, 48000), tone(440, 1.0, 24000)]
        joined = scrivo.join_capture_blocks(blocks, [48000.0, 48000.0, 24000.0], 24000.0)
        self.assertEqual(len(joined), 48000)                 # 2.0 s at 24 kHz
        self.assertAlmostEqual(dominant_hz(joined[:24000], 24000), 440, delta=5)
        self.assertAlmostEqual(dominant_hz(joined[24000:], 24000), 440, delta=5)


class RouteChangeMidCaptureTest(unittest.TestCase):
    def test_batch_audio_keeps_true_duration_and_pitch(self):
        node = FakeNode(rate=48000.0)
        engine = FakeEngine(node)
        capture = scrivo.CaptureService()
        capture._engine_obj = engine
        with fake_avfoundation(lambda: engine), \
             mock.patch.object(capture_module, "log", lambda msg: None), \
             mock.patch.object(capture_module, "_pin_input_to_builtin", lambda n: 77), \
             mock.patch.object(scrivo.threading, "Thread", InlineThread):
            capture._start_engine()                          # built-in mic, 48 kHz
            capture.begin()                                  # key down
            feed(capture, tone(440, 1.0, 48000), 48000)      # 1 s spoken at 48 kHz
            node.rate = 24000.0                              # headset connects:
            capture._engine_started_at -= 5
            capture._on_config_change()                      # real rebuild path, now 24 kHz
            self.assertFalse(capture._route_rebuilding)
            self.assertEqual(node.calls[-1], "installTap@24000")
            feed(capture, tone(440, 1.0, 24000), 24000)      # 1 s spoken at 24 kHz
            raw = capture.end()
        rate = capture.native_rate                           # what on_finish enqueues
        self.assertEqual(rate, 24000.0)
        self.assertAlmostEqual(len(raw) / rate, 2.0, delta=0.05)
        whisper_in = scrivo.prepare_for_whisper(raw, rate)
        third = len(whisper_in) // 3
        self.assertAlmostEqual(dominant_hz(whisper_in[:third], 16000), 440, delta=10)
        self.assertAlmostEqual(dominant_hz(whisper_in[-third:], 16000), 440, delta=10)


# -- F45: a failed microphone start surfaces ----------------------------------

def run_closures(names, namespace):
    """Compile the named functions nested directly inside scrivo.run() verbatim
    (same file, same line numbers) into `namespace`, whose entries stand in
    for the closure variables they read. This exercises the real start /
    finish / discard wiring without a menu bar, an event tap or a mic."""
    tree = ast.parse(Path(scrivo.__file__).read_text(encoding="utf-8"))
    run = next(node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name == "run")
    found = {node.name: node for node in run.body
             if isinstance(node, ast.FunctionDef) and node.name in names}
    missing = [name for name in names if name not in found]
    if missing:
        raise AssertionError(f"scrivo.run() defines no closure(s) {missing}")
    module = ast.Module(body=[found[name] for name in names], type_ignores=[])
    exec(compile(module, scrivo.__file__, "exec"), namespace)
    return namespace


class StatusUI:
    """Records every overlay call as (method, *args)."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args: self.calls.append((name, *args))

    def names(self):
        return [call[0] for call in self.calls]


class RecordingThread(threading.Thread):
    """Every thread the code under test starts, so the test can join them."""
    started = []

    def start(self):
        RecordingThread.started.append(self)
        super().start()


class FailedMicStartTest(unittest.TestCase):
    def setUp(self):
        RecordingThread.started = []
        self.logs = []
        real_sleep = time.sleep
        for patcher in (mock.patch.object(scrivo.threading, "Thread", RecordingThread),
                        mock.patch.object(scrivo.time, "sleep", lambda s: None),  # retry pauses
                        mock.patch.object(capture_module, "log", self.logs.append)):
            patcher.start()
            self.addCleanup(patcher.stop)
        # announce_live's own clock: its 10 s "mic live" deadline passes in 0.5 s
        self.fast_time = types.SimpleNamespace(monotonic=lambda: time.monotonic() * 20,
                                               sleep=real_sleep, time=time.time)

    def wire(self, *, with_handler, ui_call=None, failure_message=None):
        capture = scrivo.CaptureService()
        capture._engine_obj = FakeEngine(start_ok=False)   # device busy or gone
        ui, jobs = StatusUI(), queue.Queue()

        @contextlib.contextmanager
        def starting():
            yield True

        namespace = {
            "shutdown": types.SimpleNamespace(requested=lambda: False,
                                              enqueue=lambda q, job: q.put(job) or True,
                                              stop_capture=lambda c: None),
            "capture_gate": types.SimpleNamespace(starting=starting),
            "pending_deliveries": scrivo.PendingDeliveries(), "finishing": scrivo.PendingDeliveries(),
            "use_nemotron": False, "capture": capture, "jobs": jobs,
            "model_rewarm_due": lambda *a, **k: False,
            "model_activity": {"last_finished": 0.0, "rewarming": False},
            "adaptive": False, "status_ui": ui,
            "ui_call": ui_call or (lambda method, *a: method(*a)),
            "log": self.logs.append, "time": self.fast_time, "threading": threading,
            "uuid": uuid, "current_speech_config": lambda: None, "np": np,
            "spool": types.SimpleNamespace(submit=lambda *a: None),  # N9b; tests/test_capture_spool.py
            "mic_alert": threading.Lock(),
            "mic_start_failure_message": lambda: failure_message or capture_module.MIC_START_FAILED_TEXT,
        }
        names = ["on_start", "on_finish", "finish_capture", "on_discard"]
        if with_handler:
            names.append("on_mic_failed")
        run_closures(names, namespace)
        gesture = scrivo.GestureEngine(namespace["on_start"], namespace["on_finish"],
                                      namespace["on_discard"])
        namespace["engine"] = gesture
        if with_handler:
            capture.on_start_failed = namespace["on_mic_failed"]
        return capture, gesture, ui, jobs

    def join_all(self):
        """Wait for every thread started so far, including ones they start."""
        seen = 0
        while seen < len(RecordingThread.started):
            batch = RecordingThread.started[seen:]
            seen = len(RecordingThread.started)
            for thread in batch:
                thread.join(10)
                self.assertFalse(thread.is_alive())

    def test_a_failed_start_never_announces_a_live_mic(self):
        with fake_avfoundation(lambda: None), \
             mock.patch.object(capture_module, "_pin_input_to_builtin", lambda node: 77):
            capture, gesture, ui, jobs = self.wire(with_handler=False)
            gesture.pressed()                               # key down: five failed starts
            self.join_all()
        self.assertIn("! could not start the mic after 5 tries", self.logs)
        self.assertNotIn("● recording (mic live)", self.logs)
        after_waking = ui.names()[ui.names().index("show_waking") + 1:]
        self.assertNotIn("show_recording", after_waking, "the orb went solid on a dead mic")
        self.assertFalse(capture.is_waking())

    def test_a_failed_start_ends_the_gesture_and_tells_the_user_once(self):
        with fake_avfoundation(lambda: None), \
             mock.patch.object(capture_module, "_pin_input_to_builtin", lambda node: 77):
            capture, gesture, ui, jobs = self.wire(with_handler=True)
            gesture.pressed()
            self.join_all()
            self.assertEqual(gesture.snapshot(), (False, False), "gesture kept recording a dead mic")
            self.assertFalse(capture.is_active())
            gesture.released()                              # the key-up finds nothing to finish
            self.join_all()
        errors = [call for call in ui.calls if call[0] == "show_error"]
        self.assertEqual(len(errors), 1, ui.calls)
        self.assertEqual(errors[0][1], "Could not start the microphone")
        self.assertIn("hide", ui.names()[ui.names().index("show_waking") + 1:])  # the orb goes away
        self.assertNotIn("● recording (mic live)", self.logs)
        self.assertEqual(jobs.qsize(), 0)                   # nothing pretends to record
        self.assertLessEqual(sum("captured" in line for line in self.logs), 1)

    def press_and_release(self, gesture):
        gesture.pressed()
        self.join_all()
        gesture.released()
        self.join_all()

    def test_a_press_while_the_alert_is_open_does_not_stack_a_second_one(self):
        """2026-10-25: two presses during a mic outage stacked two identical
        alerts, because callAfter runs inside the open alert's modal loop."""
        posted = []

        def run_posted():
            while posted:
                method, args = posted.pop(0)
                method(*args)

        with fake_avfoundation(lambda: None), \
             mock.patch.object(capture_module, "_pin_input_to_builtin", lambda node: 77):
            capture, gesture, ui, jobs = self.wire(
                with_handler=True, ui_call=lambda method, *a: posted.append((method, a)))
            self.press_and_release(gesture)    # its alert is posted but still open
            self.press_and_release(gesture)
            run_posted()                       # the user dismisses it
            self.assertEqual(ui.names().count("show_error"), 1, ui.calls)
            self.assertIn("○ microphone alert already open — not shown again", self.logs)
            self.press_and_release(gesture)    # a later failure is reported again
            run_posted()
        self.assertEqual(ui.names().count("show_error"), 2, ui.calls)

    def test_the_alert_says_what_the_failure_check_found(self):
        with fake_avfoundation(lambda: None), \
             mock.patch.object(capture_module, "_pin_input_to_builtin", lambda node: 77):
            capture, gesture, ui, jobs = self.wire(with_handler=True,
                                                   failure_message="Python (mac_run.py) holds it")
            self.press_and_release(gesture)
        errors = [call for call in ui.calls if call[0] == "show_error"]
        self.assertEqual(errors, [("show_error", "Could not start the microphone",
                                   "Python (mac_run.py) holds it")])

    def test_an_alert_that_could_not_be_posted_does_not_silence_later_ones(self):
        refused = []

        def ui_call(method, *args):
            if method.__name__ == "show_mic_alert" and not refused:
                refused.append(method)
                raise RuntimeError("callAfter failed")
            method(*args)

        with fake_avfoundation(lambda: None), \
             mock.patch.object(capture_module, "_pin_input_to_builtin", lambda node: 77):
            capture, gesture, ui, jobs = self.wire(with_handler=True, ui_call=ui_call)
            self.press_and_release(gesture)    # its alert never reaches the main thread
            self.press_and_release(gesture)
        self.assertEqual(len(refused), 1)
        self.assertEqual(ui.names().count("show_error"), 1, ui.calls)
        self.assertIn("! mic failure handler failed: callAfter failed", self.logs)


# -- 2026-10-25: no lockable memory left for audio -----------------------------

class MicStartFailureMessageTest(unittest.TestCase):
    """A local model run held 48.8 GB of Metal memory (wired) on a 64 GB Mac.
    Wired memory passed vm.global_user_wire_limit, coreaudiod's mlock of the
    mic's I/O buffer failed with EAGAIN, and kAUStartIO failed for every app.
    The alert blamed permissions; it should name what holds the memory."""

    def message(self, *, can_lock, holder=(50135, 48_800_000_000)):
        with mock.patch.object(capture_module, "_can_lock_audio_buffer", lambda: can_lock), \
             mock.patch.object(capture_module, "_largest_wired_process", lambda: holder), \
             mock.patch.object(capture_module, "_process_label",
                               lambda pid: f"Python (mac_run.py, pid {pid})"):
            return capture_module.mic_start_failure_message()

    def test_no_lockable_memory_names_the_process_holding_it(self):
        text = self.message(can_lock=False)
        self.assertIn("Python (mac_run.py, pid 50135) is holding 48.8 GB", text)
        self.assertIn("no app can use the microphone", text)
        self.assertNotIn("Privacy & Security", text)

    def test_lockable_memory_keeps_the_device_and_permission_advice(self):
        text = self.message(can_lock=True)
        self.assertEqual(text, capture_module.MIC_START_FAILED_TEXT)
        self.assertIn("Privacy & Security", text)

    def test_a_small_holder_is_not_blamed(self):
        text = self.message(can_lock=False, holder=(4242, 300_000_000))
        self.assertIn("no app can use the microphone", text)
        self.assertNotIn("4242", text)

    def test_no_inspectable_holder_still_explains_the_shortage(self):
        text = self.message(can_lock=False, holder=None)
        self.assertIn("no app can use the microphone", text)

    def test_a_failed_check_falls_back_to_the_device_advice(self):
        def unavailable():
            raise OSError(1, "mlock unavailable")

        logs = []
        with mock.patch.object(capture_module, "_can_lock_audio_buffer", unavailable), \
             mock.patch.object(capture_module, "log", logs.append):
            self.assertEqual(capture_module.mic_start_failure_message(),
                             capture_module.MIC_START_FAILED_TEXT)
        self.assertEqual(logs, ["! memory check failed: [Errno 1] mlock unavailable"])

    @unittest.skipUnless(sys.platform == "darwin", "calls macOS mlock")
    def test_a_finite_lock_limit_of_our_own_proves_nothing(self):
        """mlock's EAGAIN also means this process hit its own RLIMIT_MEMLOCK,
        which says nothing about coreaudiod."""
        import resource
        with mock.patch.object(resource, "getrlimit", lambda which: (1024, 1024)):
            self.assertTrue(capture_module._can_lock_audio_buffer())

    @unittest.skipUnless(sys.platform == "darwin", "calls macOS mlock and libproc")
    def test_the_real_calls_run_on_this_mac(self):
        self.assertIsInstance(capture_module._can_lock_audio_buffer(), bool)
        holder = capture_module._largest_wired_process()
        self.assertTrue(holder is None or (holder[0] > 0 and holder[1] >= 0), holder)
        self.assertIn(f"pid {os.getpid()})", capture_module._process_label(os.getpid()))


# -- F41: doctor probes the input the capture path records from ---------------

class DoctorProbeTest(unittest.TestCase):
    def test_probe_pins_the_capture_mic_then_reads_the_input_bus(self):
        node = FakeNode(rate=48000.0)
        engine = FakeEngine(node)
        pinned = []

        def pin(target_node):
            pinned.append(target_node)
            node.calls.append("pin")
            return 77

        out = __import__("io").StringIO()
        with fake_avfoundation(lambda: engine), \
             mock.patch.object(capture_module, "_pin_input_to_builtin", pin), \
             contextlib.redirect_stdout(out):
            status = scrivo.doctor_input_probe()
        self.assertEqual(status, 0)
        self.assertEqual(pinned, [node])
        self.assertEqual(node.calls, ["pin", "inputFormat"],
                         "probe must pin first, then read the INPUT bus — never the output bus")
        self.assertEqual(out.getvalue().strip(), "48000 Hz, 1 ch")

    @unittest.skipUnless(sys.platform == "darwin",
                         "macOS doctor: patches ApplicationServices.AXIsProcessTrusted")
    def test_doctor_runs_the_probe_through_scrivo_not_an_unpinned_output_bus(self):
        logs = []
        completed = types.SimpleNamespace(returncode=0, stdout="48000 Hz, 1 ch\n",
                                          stderr="  mic: built-in mic\n")
        with mock.patch("subprocess.run", return_value=completed) as run, \
             mock.patch.object(scrivo, "microphone_permission", return_value=True), \
             mock.patch("ApplicationServices.AXIsProcessTrusted", return_value=True), \
             mock.patch.object(scrivo, "log", logs.append):
            scrivo.doctor()
        command = run.call_args.args[0]
        self.assertEqual(command[0], sys.executable)
        probe = " ".join(command)
        self.assertIn("doctor_input_probe", probe)
        self.assertNotIn("outputFormatForBus_", probe)
        self.assertEqual(run.call_args.kwargs.get("cwd"), str(Path(scrivo.__file__).resolve().parent))
        self.assertIn("✓ input device: 48000 Hz, 1 ch (built-in mic)", logs)


# -- which mic a key press records from, pinned before capture moves out of scrivo.py

def device(dev_id, name, transport, inputs=(0x0201,), outputs=()):
    return {"id": dev_id, "name": name, "transport": fourcc(transport),
            "input_terminals": list(inputs), "output_terminals": list(outputs)}


BUILT_IN = device(1, "MacBook Pro Microphone", "bltn")


class DeviceSelectionTest(unittest.TestCase):
    def select(self, devices):
        with mock.patch.object(capture_module, "_audio_devices", lambda: devices):
            return capture_module._select_input_device()

    def test_a_worn_headset_beats_the_built_in_mic(self):
        airpods = device(2, "Zohar's AirPods Pro", "blue")
        self.assertEqual(self.select([BUILT_IN, airpods]), (2, "Zohar's AirPods Pro (headset mic)"))

    def test_headset_terminal_types_mark_a_worn_mic(self):
        for name, inputs, outputs in (("USB Audio", (0x0402,), ()), ("USB Audio", (0x0201,), (0x0302,)),
                                      ("USB Audio", (fourcc("hmic"),), ()),
                                      ("USB Audio", (0x0201,), (fourcc("hdph"),))):
            with self.subTest(inputs=inputs, outputs=outputs):
                usb = device(3, name, "usb ", inputs, outputs)
                self.assertEqual(self.select([BUILT_IN, usb]), (3, "USB Audio (headset mic)"))

    def test_continuity_and_virtual_inputs_are_never_chosen(self):
        iphone = device(4, "iPhone headset Microphone", "ccwd", (0x0402,))
        loopback = device(5, "Loopback headset", "virt", (0x0402,))
        self.assertEqual(self.select([iphone, loopback, BUILT_IN]), (1, "built-in mic"))

    def test_a_room_speaker_mic_loses_to_the_built_in(self):
        speaker = device(6, "JBL Flip 6", "blue", (0x0201,), (0x0301,))
        self.assertEqual(self.select([speaker, BUILT_IN]), (1, "built-in mic"))

    def test_the_first_worn_headset_and_the_first_built_in_win(self):
        first, second = device(7, "Beats Flex", "blue"), device(8, "Wired EarPods", "usb ")
        self.assertEqual(self.select([first, second, BUILT_IN]), (7, "Beats Flex (headset mic)"))
        later_built_in = device(9, "Second built-in", "bltn")
        self.assertEqual(self.select([BUILT_IN, later_built_in]), (1, "built-in mic"))

    def test_no_usable_device_and_a_failed_query_say_why(self):
        self.assertEqual(self.select([device(4, "iPhone Microphone", "ccwd")]),
                         (None, "no suitable input device"))

        def broken():
            raise OSError("CoreAudio gone")
        with mock.patch.object(capture_module, "_audio_devices", broken):
            self.assertEqual(capture_module._select_input_device(), (None, "device query failed"))

    def test_the_built_in_failover_target_ignores_a_worn_headset(self):
        airpods = device(2, "AirPods", "blue")
        with mock.patch.object(capture_module, "_audio_devices", lambda: [airpods, BUILT_IN]):
            self.assertEqual(capture_module._builtin_input_device(), 1)
        with mock.patch.object(capture_module, "_audio_devices", lambda: [airpods]):
            self.assertIsNone(capture_module._builtin_input_device())

        def broken():
            raise OSError("CoreAudio gone")
        with mock.patch.object(capture_module, "_audio_devices", broken):
            self.assertIsNone(capture_module._builtin_input_device())


class FakeAudioToolbox:
    def __init__(self, status=0):
        self.status = status
        self.calls = []

    def AudioUnitSetProperty(self, unit, prop, scope, element, data_ref, size):
        self.calls.append((unit.value, prop, scope, element, data_ref._obj.value, size))
        return self.status


class PinnableNode:
    """Just enough of AVAudioInputNode for _pin_input_to_builtin."""
    UNIT = 0xA0D10

    def audioUnit(self):
        return types.SimpleNamespace(pointerAsInteger=self.UNIT)


class PinInputTest(unittest.TestCase):
    def pin(self, selected, toolbox):
        logs, opened = [], []

        def cdll(path, *args, **kwargs):
            opened.append(path)
            if isinstance(toolbox, Exception):
                raise toolbox
            return toolbox

        with mock.patch.object(capture_module, "_select_input_device", lambda: selected), \
             mock.patch.object(capture_module, "log", logs.append), \
             mock.patch("ctypes.CDLL", cdll):
            result = capture_module._pin_input_to_builtin(PinnableNode())
        return result, logs, opened

    def test_the_selected_device_becomes_the_input_units_current_device(self):
        toolbox = FakeAudioToolbox()
        result, logs, opened = self.pin((2, "AirPods (headset mic)"), toolbox)
        self.assertEqual(result, 2)
        self.assertEqual(logs, ["  mic: AirPods (headset mic)"])
        self.assertEqual(opened, ["/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox"])
        self.assertEqual(toolbox.calls, [(PinnableNode.UNIT, 2000, 0, 0, 2, 4)])

    def test_a_rejected_pin_still_reports_the_intended_route(self):
        result, logs, _ = self.pin((1, "built-in mic"), FakeAudioToolbox(status=-10851))
        self.assertEqual(result, 1)
        self.assertEqual(logs, ["  mic: built-in mic",
                                "! mic pin rejected (err -10851) — using system default input"])

    def test_a_pin_that_raises_falls_back_to_the_default_input(self):
        result, logs, _ = self.pin((1, "built-in mic"), OSError("no AudioToolbox"))
        self.assertEqual(result, 1)
        self.assertEqual(logs, ["  mic: built-in mic",
                                "! could not pin the mic — using system default input"])

    def test_no_device_pins_nothing(self):
        result, logs, opened = self.pin((None, "no suitable input device"), FakeAudioToolbox())
        self.assertIsNone(result)
        self.assertEqual((logs, opened), ([], []))


class IdleDeviceScanTest(unittest.TestCase):
    """tick() follows a plugged-in or removed headset only while idle."""

    def setUp(self):
        self.logs = []
        patcher = mock.patch.object(capture_module, "log", self.logs.append)
        patcher.start()
        self.addCleanup(patcher.stop)

    def idle_capture(self, wanted):
        capture = scrivo.CaptureService()
        engine = FakeEngine()
        engine.running = True
        capture._engine_obj = capture._engine = engine
        capture._node = engine.node
        capture._device_id = 1
        capture.idle_release_s = -1                          # keep the engine; only the scan acts
        calls = []
        patchers = (mock.patch.object(capture_module, "_select_input_device", lambda: wanted),
                    mock.patch.object(capture, "_release_engine", lambda **kw: calls.append("release") or True),
                    mock.patch.object(capture, "_start_engine", lambda: calls.append("start")))
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        return capture, calls

    def test_a_new_device_rebuilds_the_engine_on_it(self):
        capture, calls = self.idle_capture((2, "AirPods (headset mic)"))
        capture.tick()
        self.assertEqual(calls, ["release", "start"])
        self.assertIn("! input device changed — switching to AirPods (headset mic)", self.logs)

    def test_the_same_device_or_a_recent_scan_changes_nothing(self):
        capture, calls = self.idle_capture((1, "built-in mic"))
        capture.tick()
        self.assertGreater(capture._last_device_scan, 0.0)
        capture._device_id = 3                               # would switch, but the scan just ran
        capture.tick()
        self.assertEqual(calls, [])
        self.assertEqual(self.logs, [])

    def test_no_scan_while_recording(self):
        capture, calls = self.idle_capture((2, "AirPods (headset mic)"))
        capture.begin()
        capture.tick()
        self.assertEqual(calls, [])
        self.assertEqual(capture._last_device_scan, 0.0)


if __name__ == "__main__":
    unittest.main()
