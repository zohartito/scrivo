"""Key-scoped microphone capture: CaptureService and its CoreAudio helpers.

Key-down starts AVAudioEngine on the chosen mic (a worn headset, else the
built-in) and installs the input tap; key-up stops the engine and removes the
tap, so idle Scrivo holds no microphone. The service also owns the idle device
scan, the dead-route repair ladder (drop the reused engine object, then a
launchd relaunch) and joining tap blocks recorded at different rates.

Moved verbatim out of scrivo.py (restructure step 4); scrivo re-exports the
public names, so ``scrivo.CaptureService is capture.CaptureService``. The
service reads its helpers, constants and log from this module: patch them here
(``capture._pin_input_to_builtin``, ``capture.log``), never through scrivo.
Imports nothing from scrivo.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from scrivo_log import log


SAMPLE_RATE = 16_000
SILENCE_RESTART_AFTER_S = 600.0  # min process age before a dead-mic restart


# USB-audio terminal types macOS reports for streams (CoreAudio also uses
# fourcc forms). A headset mic sits near the mouth, so it beats the built-in;
# a plain speaker's mic sits across the room, so the built-in beats it.
HEADSET_INPUT_TERMINALS = {0x0402, 0x0403, 0x0205,
                           int.from_bytes(b"hmic", "big")}
HEADPHONE_OUTPUT_TERMINALS = {0x0302, 0x0402,
                              int.from_bytes(b"hdph", "big")}
HEADSET_NAME_HINTS = ("airpod", "headphone", "headset", "earbud", "earpod",
                      "buds", "beats", "wh-", "wf-")


def _audio_devices() -> list[dict]:
    """Every input-capable device with the facts needed to choose one."""
    import ctypes
    ca = ctypes.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
    cf = ctypes.CDLL(
        "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")

    def fourcc(code: str) -> int:
        return int.from_bytes(code.encode("ascii"), "big")

    class _Addr(ctypes.Structure):
        _fields_ = [("selector", ctypes.c_uint32), ("scope", ctypes.c_uint32),
                    ("element", ctypes.c_uint32)]

    def get_data(obj_id, selector, scope, buf):
        addr = _Addr(selector, scope, 0)
        size = ctypes.c_uint32(ctypes.sizeof(buf))
        err = ca.AudioObjectGetPropertyData(
            ctypes.c_uint32(obj_id), ctypes.byref(addr), 0, None,
            ctypes.byref(size), ctypes.byref(buf))
        return err, size.value

    def name_of(dev_id) -> str:
        ref = ctypes.c_void_p(0)
        get_data(dev_id, fourcc("lnam"), fourcc("glob"), ref)
        if not ref.value:
            return ""
        try:
            buf = ctypes.create_string_buffer(256)
            cf.CFStringGetCString(ref, buf, 256, 0x08000100)
            return buf.value.decode("utf-8", "replace")
        finally:
            # kAudioObjectPropertyName hands out a +1 CFStringRef that the
            # caller owns; the idle device scan runs every 5 s, so an
            # unreleased name leaked a string per device per scan.
            cf.CFRelease(ref)

    def terminals(dev_id, scope) -> list[int]:
        arr = (ctypes.c_uint32 * 32)()
        _, size = get_data(dev_id, fourcc("stm#"), scope, arr)
        found = []
        for i in range(size // 4):
            value = ctypes.c_uint32(0)
            err, _ = get_data(arr[i], fourcc("term"), fourcc("glob"), value)
            if err == 0:
                found.append(int(value.value))
        return found

    devices = (ctypes.c_uint32 * 64)()
    _, size = get_data(1, fourcc("dev#"), fourcc("glob"), devices)
    result = []
    for i in range(size // 4):
        dev_id = int(devices[i])
        transport = ctypes.c_uint32(0)
        get_data(dev_id, fourcc("tran"), fourcc("glob"), transport)
        inputs = terminals(dev_id, fourcc("inpt"))
        if not inputs:
            continue
        result.append({
            "id": dev_id,
            "name": name_of(dev_id),
            "transport": int(transport.value),
            "input_terminals": inputs,
            "output_terminals": terminals(dev_id, fourcc("outp")),
        })
    return result


def _select_input_device() -> tuple[int | None, str]:
    """Pick the mic to record from: a worn headset (AirPods, wired earbuds)
    when one is connected — its mic is at your mouth — otherwise the built-in
    MacBook mic. Never the iPhone Continuity mic or a Bluetooth speaker's mic:
    those sit across the room and produced this project's deaf recordings."""

    def fourcc(code: str) -> int:
        return int.from_bytes(code.encode("ascii"), "big")

    try:
        devices = _audio_devices()
    except Exception:
        return None, "device query failed"

    builtin = None
    headset = None
    for device in devices:
        transport = device["transport"]
        if transport in (fourcc("ccwd"), fourcc("virt")):
            continue  # iPhone Continuity / virtual (Teams, Loopback)
        if transport == fourcc("bltn") and builtin is None:
            builtin = device
            continue
        name = device["name"].lower()
        looks_worn = (
            any(t in HEADSET_INPUT_TERMINALS for t in device["input_terminals"])
            or any(t in HEADPHONE_OUTPUT_TERMINALS
                   for t in device["output_terminals"])
            or any(hint in name for hint in HEADSET_NAME_HINTS)
        )
        if looks_worn and headset is None:
            headset = device
    if headset is not None:
        return headset["id"], f"{headset['name']} (headset mic)"
    if builtin is not None:
        return builtin["id"], "built-in mic"
    return None, "no suitable input device"


def _builtin_input_device() -> int | None:
    """The MacBook's own mic, whatever is worn. This is the failover target
    when the selected headset turns out to deliver only zeros, so it must
    NOT go through _select_input_device (which prefers the headset)."""
    try:
        devices = _audio_devices()
    except Exception:
        return None
    builtin_transport = int.from_bytes(b"bltn", "big")
    for device in devices:
        if device["transport"] == builtin_transport:
            return device["id"]
    return None


def _pin_input_to_builtin(node) -> int | None:
    """Point the engine's input unit at the chosen mic (headset if worn, else
    built-in) and return that device's id. Must run BEFORE engine.prepare();
    afterwards inputFormatForBus_(0) reflects the real hardware format
    (outputFormatForBus stays stale — tap with the input one, or the engine
    silently captures nothing / refuses to start)."""
    import ctypes
    device, description = _select_input_device()
    if device is None:
        return None
    log(f"  mic: {description}")
    try:
        toolbox = ctypes.CDLL(
            "/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox")
        buf = ctypes.c_uint32(device)
        err = toolbox.AudioUnitSetProperty(
            ctypes.c_void_p(node.audioUnit().pointerAsInteger),
            2000,  # kAudioOutputUnitProperty_CurrentDevice
            0, 0, ctypes.byref(buf), 4)
        if err != 0:
            log(f"! mic pin rejected (err {err}) — using system default input")
    except Exception:
        log("! could not pin the mic — using system default input")
    # The selected device is the route we intended either way; the idle
    # device scan compares against it.
    return device


MIC_START_FAILED_TEXT = (
    "Scrivo could not open the input device. Check that another app is not holding it and "
    "that Scrivo may use the microphone (System Settings → Privacy & Security → Microphone), "
    "then press the key again.")
# macOS refuses every mlock() once the system's wired memory reaches
# vm.global_user_wire_limit (RAM minus vm.global_no_user_wire_amount, ~82%),
# and coreaudiod mlocks each client's I/O buffer when a mic starts.
# 2026-10-25: a local model benchmark held 48.8 GB of Metal memory (Metal
# memory is wired) on a 64 GB Mac; coreaudiod logged "mlock failed ... errno
# 35" and kAUStartIO returned 'what' for every app, a fresh process included,
# until that run ended. Nothing in Scrivo can start the mic then, so the
# alert names what holds the memory instead of blaming permissions.
LOCK_PROBE_BYTES = 16 * 1024            # one page; coreaudiod locked 1.2 MB per mic
WIRED_HOLDER_MIN_BYTES = 1_000_000_000  # smaller holders are not the cause


def _can_lock_audio_buffer() -> bool:
    """Whether a process may still mlock a page — the call coreaudiod makes
    when a mic starts, under the same system-wide limit. Only EAGAIN counts
    as no, and only when this process has no lock limit of its own (the
    macOS default): EAGAIN also means a finite RLIMIT_MEMLOCK was hit."""
    import ctypes
    import errno
    import resource
    if resource.getrlimit(resource.RLIMIT_MEMLOCK)[0] != resource.RLIM_INFINITY:
        return True   # cannot tell the system limit from our own
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    libc.mlock.argtypes = libc.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    buf = ctypes.create_string_buffer(LOCK_PROBE_BYTES)
    if libc.mlock(buf, len(buf)) != 0:
        return ctypes.get_errno() != errno.EAGAIN
    libc.munlock(buf, len(buf))
    return True


def _largest_wired_process() -> tuple[int, int] | None:
    """(pid, wired bytes) of the process wiring the most memory, among the
    processes this user may inspect (macOS hides system and other users')."""
    import ctypes

    class RusageInfoV0(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
            (name, ctypes.c_uint64) for name in (
                "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups",
                "pageins", "wired_size", "resident_size", "phys_footprint",
                "proc_start_abstime", "proc_exit_abstime")]

    libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    pids = (ctypes.c_int * (libproc.proc_listallpids(None, 0) + 64))()
    count = libproc.proc_listallpids(pids, ctypes.sizeof(pids))
    largest = None
    for pid in pids[:max(count, 0)]:
        usage = RusageInfoV0()
        if libproc.proc_pid_rusage(pid, 0, ctypes.byref(usage)) != 0:  # RUSAGE_INFO_V0
            continue
        if largest is None or usage.wired_size > largest[1]:
            largest = (pid, usage.wired_size)
    return largest


def _process_label(pid: int) -> str:
    """'Python (mac_run.py, pid 50135)': the program name plus the script it
    runs, so the user can tell which of several Python processes to quit."""
    import ctypes
    import subprocess
    from pathlib import Path
    buf = ctypes.create_string_buffer(256)
    ctypes.CDLL("/usr/lib/libproc.dylib").proc_name(pid, buf, len(buf))
    name = buf.value.decode("utf-8", "replace") or "a process"
    try:
        args = subprocess.run(["/bin/ps", "-o", "args=", "-p", str(pid)],
                              capture_output=True, text=True, timeout=2).stdout.split()
    except (OSError, subprocess.SubprocessError):
        args = []
    script = next((Path(arg).name for arg in args[1:] if arg.endswith(".py")), None)
    return f"{name} ({script}, pid {pid})" if script else f"{name} (pid {pid})"


def mic_start_failure_message() -> str:
    """What the mic-failure alert says. When macOS refuses to lock even a
    page no app can start a mic, so it names the biggest holder
    of locked memory; any other failure keeps the device and permission
    advice. Never raises."""
    try:
        if _can_lock_audio_buffer():
            return MIC_START_FAILED_TEXT
        holder = _largest_wired_process()
        shortage = ("Your Mac has no memory left that it can lock for audio, "
                    "so no app can use the microphone right now.")
        if holder is None or holder[1] < WIRED_HOLDER_MIN_BYTES:
            return (f"{shortage} Quit a large local AI model or another app that "
                    "uses a lot of memory, then press the key again.")
        pid, held = holder
        return (f"{shortage} {_process_label(pid)} is holding {held / 1e9:.1f} GB of it. "
                "Quit it or let it finish, then press the key again.")
    except Exception as exc:
        log(f"! memory check failed: {str(exc)[:120]}")
        return MIC_START_FAILED_TEXT


PREROLL_S = 0.35     # available only while a deliberately retained engine runs
RING_S = 5.0         # rolling pre-roll ring capacity
IDLE_STOP_S = 0.0    # production privacy boundary: stop the mic on key-up.
                     # The ASR model stays resident; AVAudioEngine does not.


class CaptureService:
    """Key-scoped capture with a process-resident ASR model.

    The daemon may stay alive and keep Whisper loaded, but the microphone does
    not: key-down starts AVAudioEngine and installs its input tap; key-up stops
    the engine and removes the tap.  The AVAudioEngine object itself is reused
    across captures because allocating a new one per press leaked CoreAudio
    threads on this machine.  Reusing an inert object does not hold the input
    device or illuminate the macOS microphone indicator.

    Buffer reading: floatChannelData()[0].as_buffer(N) takes an ELEMENT
    count, not bytes. Passing 4*n once read 4x past every buffer — real
    samples followed by stale memory — which surfaced as phantom duplicate
    delivery, glitch spikes up to 1e30, and stuttered audio that whisper
    transcribed as loops. The sampleTime dedupe below stays as a no-op
    safety net. The tap never touches ObjC-outward calls.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._ring: list[tuple[np.ndarray, float]] = []   # (block, its rate)
        self._ring_samples = 0
        self._active: list[np.ndarray] | None = None
        self._active_rates: list[float] = []   # per-block rates, parallel to _active
        self._stream = None
        self._stream_enqueue = None
        self._next_expected = None
        self._engine = None
        self._node = None
        self._starting = False
        self._start_began_at = 0.0
        self._waking = False
        self._engine_obj = None   # the one engine, reused for the process life
        self._config_observer = None
        self._engine_started_at = 0.0
        self._route_rebuilding = False
        self._zero_since = None
        self._silence_repairs = 0
        self._process_started_at = time.monotonic()
        self._restart_deferred_at = 0.0
        self.restart_callback = None
        self.on_start_failed = None   # runtime hook: the mic never came up under a capture
        self._release_thread = None
        self._last_device_scan = 0.0
        self._device_id = None
        self._last_block_at = time.monotonic()
        self._last_use = time.monotonic()
        self._closed = False
        self.idle_release_s = IDLE_STOP_S
        self.native_rate = float(SAMPLE_RATE)
        self.latest_rms = 0.0

    def _start_engine(self) -> None:
        # Starting and stopping the reusable AVAudioEngine object must never
        # overlap. A release racing the next press used to be able to stop the
        # newly started capture.
        with self._lifecycle_lock:
            started = self._start_engine_locked()
        if not started:
            self._report_start_failure()

    def _start_engine_locked(self) -> bool:
        """Build, pin, and start the capture engine. NEVER raises — an audio
        error must not kill the daemon (a stale tap format threw straight out
        of installTapOnBus and took the whole process down, which launchd then
        restarted in a loop while dictations silently captured nothing).
        Returns whether the engine is live afterwards; a False under an
        in-flight capture is surfaced by _report_start_failure.

        The format must be re-read on a FRESH engine AFTER pinning: a device
        switch (AirPods run at 24 kHz, built-in at 48 kHz) briefly leaves the
        old value in place and a mismatched tap format throws. Passing nil is
        not an option — it works on Bluetooth but fails on the built-in mic
        (both matrix-tested live). Retry until the device settles.

        Voice processing (AEC) stays OFF: on this macOS beta it corrupted
        speech into stuttered audio whisper read as loops. Music bleed is
        handled by the Silero VAD gate instead.
        """
        from AVFoundation import AVAudioEngine

        try:
            with self._lock:
                if self._closed:
                    return False
            for attempt in range(5):
                with self._lock:
                    if self._closed:
                        return False
                # ONE engine for the process lifetime. Allocating a new
                # AVAudioEngine per wake leaked its CoreAudio threads (43
                # threads / 10 audio threads observed after ~50 wake cycles,
                # burning 40% CPU and driving coreaudiod). Reuse and re-pin.
                if self._engine_obj is None:
                    self._engine_obj = AVAudioEngine.alloc().init()
                    self._observe_config_changes(self._engine_obj)
                engine = self._engine_obj
                node = engine.inputNode()
                try:
                    node.removeTapOnBus_(0)  # clear any tap from a past device
                except Exception:
                    pass
                self._device_id = _pin_input_to_builtin(node)
                if self._device_id is None:
                    log("! could not pin a mic — using system default input")
                fmt = node.inputFormatForBus_(0)
                rate = float(fmt.sampleRate())
                if rate <= 0:
                    time.sleep(0.25)
                    continue
                try:
                    node.installTapOnBus_bufferSize_format_block_(
                        0, 4096, fmt, self._tap)
                except Exception:
                    log(f"! tap rejected at {rate:.0f} Hz "
                        f"(try {attempt + 1}/5) — device still settling")
                    time.sleep(0.3)
                    continue
                self.native_rate = rate
                self._next_expected = None
                engine.prepare()
                ok, err = engine.startAndReturnError_(None)
                if not ok:
                    log(f"! mic engine failed to start: {err}")
                    try:
                        node.removeTapOnBus_(0)
                    except Exception:
                        pass
                    time.sleep(0.3)
                    continue
                with self._lock:
                    if self._closed:
                        try:
                            node.removeTapOnBus_(0)
                            engine.stop()
                        except Exception:
                            pass
                        return False
                self._engine = engine
                self._node = node
                self._last_block_at = time.monotonic()
                self._engine_started_at = time.monotonic()
                self._zero_since = None
                return True
            log("! could not start the mic after 5 tries")
            return False
        except Exception as exc:
            log(f"! mic engine error: {str(exc)[:150]}")
            return False
        finally:
            self._starting = False

    def _report_start_failure(self) -> None:
        """The engine never came up: stop pretending. Clear the waking flag
        so the overlay cannot flip to "mic live", and if a capture is waiting
        on this engine hand it to the runtime hook, which ends the gesture
        and tells the user once. Idle failures (the device scan switching
        mics) stay quiet: the next press retries and reports then."""
        with self._lock:
            self._waking = False
            pending = (self._active is not None and self._engine is None
                       and not self._closed)
        handler = self.on_start_failed
        if pending and handler is not None:
            try:
                handler()
            except Exception as exc:
                log(f"! mic failure handler failed: {str(exc)[:120]}")

    def _release_engine(self, *, only_if_idle: bool = False) -> bool:
        with self._lifecycle_lock:
            return self._release_engine_locked(only_if_idle=only_if_idle)

    def _release_engine_locked(self, *, only_if_idle: bool = False) -> bool:
        """Stop capture but KEEP the engine object — see _start_engine: a new
        AVAudioEngine per wake leaks CoreAudio threads and CPU forever.

        Every teardown step runs even when an earlier one raises: a tap
        removal that threw used to skip engine.stop(), and with the handle
        already cleared nothing ever stopped that engine — idle Scrivo kept
        the microphone open. Returns True only when the engine actually
        stopped; otherwise the handle stays so the next tick retries."""
        with self._lock:
            if only_if_idle and self._active is not None:
                return False
            engine, self._engine = self._engine, None
            node, self._node = self._node, None
            self._ring.clear()
            self._ring_samples = 0
            self._starting = False  # never leave a wake blocked behind us
        if engine is None:
            return False
        try:
            if node is not None:
                node.removeTapOnBus_(0)
        except Exception as exc:
            log(f"! mic tap not removed: {str(exc)[:120]}")
        try:
            engine.stop()
        except Exception as exc:
            log(f"! mic engine not stopped: {str(exc)[:120]}")
            with self._lock:
                if not self._closed:
                    self._engine, self._node = engine, node
            return False
        return True

    def _observe_config_changes(self, engine) -> None:
        """A route change under a live engine (dock unplugged, output device
        switched, ...) can leave it 'running' while every buffer is all
        zeros — VAD 0% on every capture until a manual restart (observed live
        2026-08-12). The system announces the change; rebuild instead of
        playing dead."""
        try:
            from AVFoundation import AVAudioEngineConfigurationChangeNotification
            from Foundation import NSNotificationCenter
            self._config_observer = (
                NSNotificationCenter.defaultCenter()
                .addObserverForName_object_queue_usingBlock_(
                    AVAudioEngineConfigurationChangeNotification, engine,
                    None, lambda note: self._on_config_change()))
        except Exception as exc:
            log(f"! route-change watch unavailable: {str(exc)[:80]}")

    def _on_config_change(self) -> None:
        with self._lock:
            # Ignore while idle (the next wake re-reads the route), while a
            # start is already in flight, and during the burst of
            # notifications our own re-pinning fires right after a start.
            if (self._closed or self._engine is None or self._starting
                    or self._route_rebuilding
                    or time.monotonic() - self._engine_started_at < 2.0):
                return
            self._route_rebuilding = True
        log("! audio route changed — rebuilding mic")

        def work() -> None:
            try:
                self._release_engine()
                with self._lock:
                    self._starting = True
                    self._start_began_at = time.monotonic()
                self._start_engine()
            finally:
                self._route_rebuilding = False

        threading.Thread(target=work, daemon=True).start()

    def tick(self) -> None:
        """Poller hook, once per second: restart an unexpectedly-dead engine
        while recording; release the mic (indicator off) after idle."""
        try:
            with self._lock:
                if self._closed:
                    return
                active = self._active is not None
                idle_for = time.monotonic() - self._last_use
                engine = self._engine
                starting = self._starting
            if engine is None or starting:
                return
            stalled = time.monotonic() - self._last_block_at > 1.5
            if (self._zero_since is not None and self._silence_repairs < 2
                    and time.monotonic() - self._zero_since > 3.0):
                self._drop_dead_capture()
                self._repair_dead_route(capturing=True)
                return
            if active and (not engine.isRunning() or stalled):
                reason = "stopped by the system" if not engine.isRunning() \
                    else "stopped delivering audio"
                log(f"! mic {reason} — restarting mid-recording")
                self._release_engine()
                self._start_engine()
                return
            elif not active and time.monotonic() - self._last_device_scan > 5.0:
                # Plugging in AirPods (or unplugging them) should switch the
                # mic without a restart. Only while idle — never mid-recording —
                # and only every 5s: a full CoreAudio enumeration costs ~4ms.
                self._last_device_scan = time.monotonic()
                wanted, description = _select_input_device()
                if wanted is not None and wanted != self._device_id:
                    log(f"! input device changed — switching to {description}")
                    self._release_engine()
                    self._start_engine()
                    return
            # Negative = never, exactly as release_soon reads it. only_if_idle:
            # "idle" was decided under the lock a moment ago, and a press that
            # lands in between must keep the engine it is about to record on.
            if (not active and self.idle_release_s >= 0
                    and idle_for > self.idle_release_s):
                if self._release_engine(only_if_idle=True):
                    log("○ mic released (idle) — wakes on next press")
        except Exception as exc:
            log(f"! mic health check failed: {str(exc)[:120]}")

    def _repair_dead_route(self, *, capturing: bool) -> None:
        """Escalating repair for a capture path delivering bit-exact zeros.

        A live mic always carries a noise floor, so exact zeros mean the route
        is dead — not that the room is quiet. 2026-09-20: AirPods dropped and
        reconnected under a new CoreAudio device id, and from then on EVERY
        capture in the running process came back bit-exact zero — including
        ones pinned to the built-in mic — while ffmpeg recorded those same
        AirPods at peak 18609 and a fresh python process running this very
        class captured normally. The device was fine; this process's input
        unit was wedged.

        Stopping and starting the engine does not clear it (the 11:03 log
        rebuilt straight back onto zeros), because the AVAudioEngine object
        is reused for the process lifetime and carries the wedged unit with
        it. So the first repair throws that object away, and the second
        throws the process away.

        Each rung fires once per silence run: the counter resets only when
        real audio arrives (_tap), never on engine start, or a genuinely
        muted input would rebuild forever."""
        if self._silence_repairs == 0:
            self._silence_repairs = 1
            log("! mic delivering only silence — rebuilding the audio engine")
            if capturing:
                self._rebuild_engine(start=True)
            else:
                # Key-up runs straight off the Quartz gesture tap, and
                # CoreAudio teardown can block it long enough for macOS to
                # disable the tap (2026-09-15). Defer it exactly like
                # release_soon does. Never start an engine here either: idle
                # scrivo must not hold the microphone (--idle-release 0), so
                # the next press is what builds the fresh one.
                threading.Thread(target=self._rebuild_engine,
                                 daemon=True).start()
            return
        # Consume the last rung only when a relaunch actually goes out. The
        # age guard can refuse one, and burning the rung there would leave a
        # deaf process with nothing left to try until real audio arrives —
        # which is exactly what never happens on a dead route.
        if self._restart_app():
            self._silence_repairs = 2

    def _rebuild_engine(self, start: bool = False) -> None:
        """Drop the reused AVAudioEngine, optionally starting a fresh one."""
        self._release_engine()
        with self._lifecycle_lock:
            self._forget_engine_object()
        if start:
            self._start_engine()

    def _forget_engine_object(self) -> None:
        """Drop the reused AVAudioEngine so the next start allocates a fresh
        one, taking its wedged HAL input unit with it. Reusing the object is
        the rule (a new engine per wake leaked CoreAudio threads); one
        allocation per dead route is not the per-wake churn that rule
        prevents."""
        engine, self._engine_obj = self._engine_obj, None
        observer, self._config_observer = self._config_observer, None
        if observer is not None:
            try:
                from Foundation import NSNotificationCenter
                NSNotificationCenter.defaultCenter().removeObserver_(observer)
            except Exception as exc:
                log(f"! route-change watch not removed: {str(exc)[:80]}")
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

    def _restart_app(self) -> bool:
        """Last rung: launchd relaunches a fresh process. A fresh process is
        the one thing observed to cure this (2026-09-20). Only for a process
        old enough that restarting cannot become a loop — a young one just
        came back from a restart, so if it is still deaf the restart is not
        the cure and the user needs to know.

        Returns whether a relaunch was actually kicked off; a refusal leaves
        the rung armed for when the process IS old enough."""
        now = time.monotonic()
        if now - self._process_started_at < SILENCE_RESTART_AFTER_S:
            # The caller retries every few seconds until then — say it once a
            # minute, not once a tick.
            if now - self._restart_deferred_at > 60.0:
                self._restart_deferred_at = now
                log("! mic still silent in a freshly started process — check "
                    "the mic in System Settings; 'Restart Scrivo' is in the menu")
            return False
        log("! mic still silent after a fresh audio engine — restarting Scrivo")
        try:
            if self.restart_callback is None:
                log("! no runtime restart handler; quit and launch Scrivo again")
                return False
            return bool(self.restart_callback())
        except Exception as exc:
            log(f"! restart failed: {str(exc)[:120]}")
            return False

    def _drop_dead_capture(self) -> None:
        """Throw away an in-flight capture that is bit-exact zeros end to end.

        Those frames came from the route we are failing over from, and a
        capture that opens with digital silence is exactly what Whisper turns
        into a repetition loop. Only drop when EVERY frame so far is zero:
        audio recorded before a live route died is real speech and must
        survive the rebuild."""
        with self._lock:
            if self._active is None:
                return
            if all(not block.any() for block in self._active):
                self._active = []
                self._active_rates = []

    def _tap(self, buffer, when) -> None:
        n = int(buffer.frameLength())
        if n <= 0:
            return
        start = int(when.sampleTime())
        ptr = buffer.floatChannelData()[0]
        data = np.frombuffer(ptr.as_buffer(n), dtype=np.float32)
        if self._next_expected is not None and start < self._next_expected:
            skip = self._next_expected - start
            if skip >= n:
                return  # pure duplicate buffer
            data = data[skip:]
            start += skip
        self._next_expected = start + len(data)
        block = data.copy()
        self.latest_rms = float(np.sqrt(np.mean(block**2)))
        # Exactly-zero buffers are a dead route, not a quiet room — a live
        # mic always carries noise floor. Timed here, acted on in tick().
        if self.latest_rms == 0.0:
            if self._zero_since is None:
                self._zero_since = time.monotonic()
        else:
            self._zero_since = None
            # Real audio: the route recovered, so the repair ladder rearms.
            # This is the ONLY place the counter resets — resetting it on
            # engine start instead made a permanently silent input rebuild
            # forever.
            self._silence_repairs = 0
        self._waking = False  # audio is flowing now
        self._last_block_at = time.monotonic()
        # trust the buffer's own rate over the format read at start-up
        try:
            self.native_rate = float(buffer.format().sampleRate())
        except Exception:
            pass
        rate = self.native_rate
        with self._lock:
            if self._active is not None:
                # Each block keeps its own rate: a route change mid-recording
                # (built-in 48 kHz -> AirPods 24 kHz) rebuilds the engine
                # under a live capture, and end() must not read the earlier
                # blocks at the later device's rate.
                self._active.append(block)
                self._active_rates.append(rate)
                if self._stream is not None:
                    # Only queue references on the tap. Resampling and ASR run
                    # on the existing serialized transcription worker.
                    self._stream_enqueue(("stream-audio", self._stream, block, rate))
            else:
                self._ring.append((block, rate))
                self._ring_samples += len(block)
                limit = int(RING_S * rate)
                while self._ring_samples > limit and len(self._ring) > 1:
                    self._ring_samples -= len(self._ring.pop(0)[0])

    def begin(self, *, stream=None, enqueue=None) -> None:
        cold_start = False
        with self._lock:
            if self._closed:
                return False
            if self._stream is not None:
                self._stream.cancelled.set()
                self._stream_enqueue(("stream-close", self._stream))
            preroll: list[np.ndarray] = []
            preroll_rates: list[float] = []
            needed = int(PREROLL_S * self.native_rate)
            collected = 0
            for block, rate in reversed(self._ring):
                preroll.insert(0, block)
                preroll_rates.insert(0, rate)
                collected += len(block)
                if collected >= needed:
                    break
            self._active = preroll
            self._active_rates = preroll_rates
            self._stream, self._stream_enqueue = stream, enqueue
            if stream is not None:
                for block, rate in zip(preroll, preroll_rates):
                    enqueue(("stream-audio", stream, block, rate))
            self._last_use = time.monotonic()
            # A start that never completed must not block every future wake —
            # treat a stale "starting" flag as dead and try again.
            stale_start = (self._starting and
                           time.monotonic() - self._start_began_at > 8.0)
            if self._engine is None and (not self._starting or stale_start):
                if stale_start:
                    log("! previous mic start never finished — retrying")
                self._starting = True
                self._start_began_at = time.monotonic()
                self._waking = True
                cold_start = True
        if cold_start:  # wake from idle — off-thread, the tap must never wait
            threading.Thread(target=self._start_engine, daemon=True).start()
        return cold_start

    def is_waking(self) -> bool:
        """True until audio actually flows after a cold start (up to ~5s)."""
        return self._waking

    def is_live(self) -> bool:
        """An engine is up; False after a start that failed every try."""
        with self._lock:
            return self._engine is not None

    def end(self, *, include_stream=False):
        with self._lock:
            frames, self._active = self._active or [], None
            rates, self._active_rates = self._active_rates, []
            stream, self._stream = self._stream, None
            self._stream_enqueue = None
            self._last_use = time.monotonic()
        if not frames:
            empty = np.zeros(0, dtype=np.float32)
            return (empty, stream) if include_stream else empty
        # Blocks placed directly (tests, old callers) carry no rate: they are
        # at the current one, which is also what the caller reads next.
        rates = rates + [self.native_rate] * (len(frames) - len(rates))
        samples = join_capture_blocks(frames, rates, self.native_rate)
        # A hold shorter than the in-capture silence watch (3s) ends before
        # tick() can react, so short presses would keep landing on the same
        # wedged input unit forever (2026-09-20: 1.37s and 1.88s holds, peak
        # 0, "ощ" x223). Repair at key-up too and the NEXT press is clean.
        # One second of exact zeros is past any Bluetooth warm-up blip and
        # cannot come from a live mic.
        if (not samples.any() and self._silence_repairs < 2
                and len(samples) / max(self.native_rate, 1.0) >= 1.0):
            self._repair_dead_route(capturing=False)
        return (samples, stream) if include_stream else samples

    def abort(self) -> None:
        with self._lock:
            if self._stream is not None:
                self._stream.cancelled.set()
                self._stream_enqueue(("stream-close", self._stream))
            self._stream = self._stream_enqueue = None
            self._active = None
            self._active_rates = []
            self._last_use = time.monotonic()

    def is_active(self) -> bool:
        with self._lock:
            return self._active is not None

    def live_text(self) -> str:
        """The streaming engine's words so far, for display only."""
        stream = self._stream
        return stream.live_text() if stream is not None and hasattr(stream, "live_text") else ""

    def release_soon(self) -> None:
        """Release the input after key-up without blocking the gesture tap.

        Production uses zero delay. A positive value remains available for
        diagnostics; a negative value deliberately retains the engine.
        """
        if self.idle_release_s < 0:
            return

        def work() -> None:
            if self.idle_release_s:
                time.sleep(self.idle_release_s)
            if self._release_engine(only_if_idle=True):
                log("○ mic released")

        threading.Thread(target=work, daemon=True).start()

    def shutdown(self) -> None:
        """Immediately release mic resources during process shutdown."""
        self.abort()
        with self._lock:
            self._closed = True
            self._active = None
            if self._release_thread is not None:
                return  # one teardown; repeated drain passes must not pile up
            # CoreAudio teardown can block; it cannot hold up the process drain.
            self._release_thread = threading.Thread(target=self._release_engine, daemon=True)
        self._release_thread.start()

    def wait_released(self, timeout: float) -> bool:
        """Bounded wait for the shutdown teardown; never blocks on a wedged HAL."""
        thread = self._release_thread
        if thread is not None:
            thread.join(timeout)
        return thread is None or not thread.is_alive()


def join_capture_blocks(frames: list, rates: list, target_rate: float) -> np.ndarray:
    """Join tap blocks into one signal at target_rate.

    A route change mid-recording (built-in mic at 48 kHz, then AirPods at
    24 kHz) leaves blocks of both rates in one capture. Reading them all at
    the last device's rate stretched the earlier speech to twice its length
    and halved its pitch; instead every run of same-rate blocks is resampled
    on its own before the runs are concatenated."""
    runs: list[tuple[list, float]] = []
    for block, rate in zip(frames, rates):
        if runs and runs[-1][1] == rate:
            runs[-1][0].append(block)
        else:
            runs.append(([block], rate))
    pieces = []
    for blocks, rate in runs:
        samples = np.concatenate(blocks).reshape(-1)
        if rate != target_rate and rate > 0 and target_rate > 0:
            from fractions import Fraction

            from scipy.signal import resample_poly
            ratio = Fraction(int(target_rate), int(rate)).limit_denominator(1000)
            samples = resample_poly(samples, ratio.numerator,
                                    ratio.denominator).astype(np.float32)
        pieces.append(samples)
    return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)


def doctor_input_probe() -> int:
    """The doctor's input check, run in a subprocess (see doctor): pin the
    same mic a key press pins, THEN read the input bus format the tap will
    use. The output bus of an unpinned engine describes the system default
    device — a USB interface at 96 kHz / 2 ch, say — while dictation records
    the pinned built-in or headset mic at its own rate."""
    from AVFoundation import AVAudioEngine
    engine = AVAudioEngine.alloc().init()
    node = engine.inputNode()
    _pin_input_to_builtin(node)   # logs "mic: <which one>" on stderr
    fmt = node.inputFormatForBus_(0)
    print(f"{fmt.sampleRate():.0f} Hz, {fmt.channelCount()} ch")
    return 0 if fmt.sampleRate() > 0 and fmt.channelCount() > 0 else 1
