"""How Scrivo stops, restarts and refuses new work while it does.

ShutdownBoundary is the one-way, event-only shutdown gate (signal handlers
only set events); CaptureGate and Lifecycle decide who may start a recording
and run the drained stop shared by Quit, the menu Restart, an installed
update and SIGINT/SIGTERM; RestartController restarts through launchd or a
drained foreground exec. The drain helpers (wait_until_idle,
dictation_in_flight, end_capture_now, main_thread_dispatch) and the exit and
drain constants live here too. ShutdownBoundary, CaptureGate and the drain
constants are shared with the Windows app.

Moved verbatim out of sotto.py (restructure step 2); sotto re-exports every
public name, so ``sotto.Lifecycle is lifecycle.Lifecycle``. Patch the launchd
probes RestartController calls (launchd_owns_this_process,
launched_as_sealed_job) here, not through sotto. Imports nothing from sotto.
"""
from __future__ import annotations

import os
import queue
import re
import signal
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from gestures import HANDS_FREE_MAX_S
from sotto_log import log

if TYPE_CHECKING:
    from delivery import PendingDeliveries


APP_DRAIN_TIMEOUT = 1.0
RESTART_DRAIN_DEADLINE_S = 20.0  # a wedged native call must not block recovery forever
QUIT_DRAIN_S = 10.0              # Quit finishes the dictation in progress, up to this long
# launchd SIGKILLs Scrivo's job 5 s after SIGTERM (`launchctl print` shows
# "exit timeout = 5"; logout, kickstart -k and rollout all send SIGTERM). The
# drain and the teardown's waits (clipboard restore, a finish short of the
# queue, an append under way, the worker join) share one deadline that leaves
# SIGTERM_KEEP_RESERVE_S to keep what is left in History before that.
LAUNCHD_EXIT_TIMEOUT_S = 5.0
SIGTERM_KEEP_RESERVE_S = 1.5
SIGTERM_DRAIN_S = 2.0
SIGTERM_RESTORE_WAIT_S = 1.0
DRAIN_POLL_S = 0.1               # how often a Quit/Restart drain re-checks the dictation
# A foreground restart that cannot exec exits with this status, never 0:
# KeepAlive {SuccessfulExit: false} (the login item, the sealed LaunchAgent)
# then starts Scrivo again instead of leaving the user without it.
RESTART_FAILED_EXIT = 75         # EX_TEMPFAIL
# An installed update restarts once dictation is done: the longest recording
# (the hands-free watchdog) plus time to transcribe and paste it, then anyway.
UPDATE_DRAIN_DEADLINE_S = HANDS_FREE_MAX_S + 120.0
RESTART_RELEASE_WAIT_S = 2.0     # bounded wait for the async mic teardown before exec


class ShutdownBoundary:
    """One-way, event-only shutdown gate for the interactive runtime.

    Signal handlers deliberately only set ``event``.  Capture/UI teardown and
    queue accounting happen on normal Python/AppKit execution paths, so a
    signal cannot interrupt a SQLite write or start a second mutation.
    """

    def __init__(self) -> None:
        self.event = threading.Event()
        # Set by the first SIGINT/SIGTERM: Lifecycle.watch_signals then stops
        # the way Quit does, finishing the dictation in flight (F1c).
        self.stop_event = threading.Event()
        self.stop_signal = None
        self.stop_at = None  # time.monotonic() of that first signal

    def requested(self) -> bool:
        return self.event.is_set()

    def request(self, *_unused) -> None:
        self.event.set()

    def request_stop(self, signum=None, _frame=None) -> None:
        """SIGINT/SIGTERM handler, event-only: the first one asks for Quit's
        drained stop; another one while that drains stops at once."""
        if self.stop_event.is_set():
            self.event.set()
            return
        self.stop_signal = signum
        self.stop_at = time.monotonic()
        self.stop_event.set()

    def teardown_wait_s(self, cap_s: float, now: float) -> float:
        """How long a teardown wait capped at ``cap_s`` may take. After
        SIGTERM every such wait shares one deadline that still leaves
        SIGTERM_KEEP_RESERVE_S before launchd's SIGKILL, so untranscribed
        captures reach History (N9); otherwise just ``cap_s``."""
        if self.stop_signal != signal.SIGTERM or self.stop_at is None:
            return cap_s
        keep_by = self.stop_at + LAUNCHD_EXIT_TIMEOUT_S - SIGTERM_KEEP_RESERVE_S
        return max(0.0, min(cap_s, keep_by - now))

    def enqueue(self, jobs: queue.Queue, job: tuple, keep=None) -> bool:
        """Queue work only while live; race with shutdown is discarded below.
        ``keep`` is handed a live capture that is refused or discarded, as in
        discard_queued, so its audio can stay in History (N9)."""
        if self.requested():
            if keep is not None and job[0] == "live":
                keep(job)
            return False
        jobs.put(job)
        if self.requested():
            self.discard_queued(jobs, keep=keep)
            return False
        return True

    @staticmethod
    def discard_queued(jobs: queue.Queue, keep=None) -> int:
        """Drain queued work with matching task accounting, never executing it.
        ``keep`` is handed each queued live capture first, so its audio can
        stay in History (N9)."""
        discarded = 0
        while True:
            try:
                job = jobs.get_nowait()
            except queue.Empty:
                return discarded
            else:
                try:
                    if keep is not None and job[0] == "live":
                        keep(job)
                finally:
                    jobs.task_done()
                discarded += 1

    def stop_capture(self, capture) -> None:
        """Discard an active recording; it must never become a history row."""
        self.request()
        try:
            capture.abort()
        finally:
            close = getattr(capture, "shutdown", None)
            if callable(close):
                close()
            else:
                capture.release_soon()


@contextmanager
def shutdown_signal_handlers(boundary: ShutdownBoundary):
    """Install event-only SIGTERM/SIGINT handlers and restore embedded callers.
    The first signal asks for a drained stop (see ShutdownBoundary.request_stop)."""
    previous: dict[int, object] = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, boundary.request_stop)
    try:
        yield boundary
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def commit_if_running(boundary: ShutdownBoundary, commit) -> bool:
    """Execute a post-inference side effect only while shutdown is clear."""
    if boundary.requested():
        return False
    commit()
    return True


LAUNCHD_LABEL = "com.zohartito.sotto.app"  # must match launchd/ and scripts/rollout.sh


def app_mode() -> bool:
    """Started by the installed Scrivo.app launcher (not a terminal)."""
    return os.environ.get("SOTTO_LAUNCHER") == "app"


def launchd_owns_this_process() -> bool:
    """An installed job with the same name must never control a source run."""
    import subprocess
    try:
        result = subprocess.run(
            ["/bin/launchctl", "print", f"gui/{os.getuid()}/{LAUNCHD_LABEL}"],
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return False
    match = re.search(r"^\s*pid = (\d+)\s*$", result.stdout, re.MULTILINE)
    return result.returncode == 0 and match is not None and int(match[1]) == os.getpid()


def launched_as_sealed_job() -> bool:
    """Does anything say launchd started this process from a sealed release?

    launchd sets XPC_SERVICE_NAME to the job label, and the sealed bootstrap
    runs ``<releases>/<digest>/source/<entry>``.  Either one means a restart
    must go through launchd: an exec here would bypass the bootstrap's
    verification and ``-I -S -B`` isolation."""
    if os.environ.get("XPC_SERVICE_NAME") == LAUNCHD_LABEL:
        return True
    entry = Path(sys.argv[0]).absolute() if sys.argv and sys.argv[0] else None
    return entry is not None and entry.parent.name == "source" and entry.parent.parent.parent.name == "releases"


class RestartController:
    """Restart our supervisor or request a drained foreground exec."""

    def __init__(self, shutdown: ShutdownBoundary):
        self.shutdown = shutdown
        self.managed = launchd_owns_this_process()
        # Fail closed: an unconfirmed launchd job never execs itself.
        self.exec_allowed = not self.managed and not launched_as_sealed_job()
        self.foreground_pending = False
        self.argv = [sys.executable, *sys.argv]
        self._on_failure = None

    def request(self, on_failure=None) -> bool:
        if self.shutdown.requested():
            return False
        if not self.managed and not self.exec_allowed:
            raise RuntimeError("Scrivo could not confirm its launchd job; quit and launch it again")
        self._on_failure = on_failure
        if self.managed:
            # launchd stops this very process during kickstart -k; keep the
            # main thread free so that SIGTERM still shuts down gracefully.
            threading.Thread(target=self._kickstart, daemon=True).start()
        else:
            self.foreground_pending = True
            self.shutdown.request()
        return True

    def _kickstart(self) -> None:
        import subprocess
        try:
            result = subprocess.run(
                ["/bin/launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{LAUNCHD_LABEL}"],
                capture_output=True, text=True, timeout=30)
            failed = result.returncode != 0
        except (OSError, subprocess.TimeoutExpired):
            failed = True
        if failed:
            self.failed("launchd could not restart Scrivo; quit and launch it again")

    def failed(self, message: str) -> None:
        """Report a restart that failed after request() already returned."""
        callback, self._on_failure = self._on_failure, None
        log(f"! restart failed: {message}")
        if callback is not None:
            callback(message)

    def exec_foreground(self) -> None:
        if not self.foreground_pending:
            return
        if app_mode() and os.environ.get("SOTTO_APP_EXECUTABLE", "").strip():
            relaunch_app_after_exit()
            return
        os.execv(sys.executable, self.argv)


APP_RELAUNCH_SCRIPT = (
    'while /bin/kill -0 "$1" 2>/dev/null; do /bin/sleep 0.1; done; '
    '/bin/launchctl kickstart "gui/$(/usr/bin/id -u)/$2" 2>/dev/null || /usr/bin/open -n "$3"')


def relaunch_app_after_exit() -> None:
    """Scrivo.app restarts as a fresh process: an in-place exec keeps the pid,
    and macOS then leaves the new menu bar icon blank and unclickable
    (2026-09-30). A detached helper waits for the launcher to exit, then
    starts the login item if it is loaded, otherwise opens the app again."""
    import subprocess
    import login_item
    bundle = Path(os.environ["SOTTO_APP_EXECUTABLE"].strip()).parents[2]
    subprocess.Popen(
        ["/bin/sh", "-c", APP_RELAUNCH_SCRIPT, "sotto-relaunch", str(os.getppid()), login_item.LABEL, str(bundle)],
        start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, close_fds=True)
    log("restarting Scrivo.app")


def main_thread_dispatch(has_ui: bool, call_after, deliveries: PendingDeliveries | None = None):
    """Return (ui_call, deliver_call), both scheduling work on the main thread.

    ui_call updates the menu bar and pill, so it does nothing under
    --no-overlay. deliver_call (pasting, voice undo) always runs: hiding the
    UI must never stop text from reaching the cursor. Each delivery is counted
    in deliveries until the scheduled method calls deliveries.finish().
    """
    def deliver_call(method, *call_args) -> None:
        if deliveries is not None:
            deliveries.add()
        call_after(method, *call_args)

    def ui_call(method, *call_args) -> None:
        if has_ui:
            call_after(method, *call_args)

    return ui_call, deliver_call


def wait_until_idle(busy, *, poll_s: float = 0.5, sleep=time.sleep,
                    deadline_s: float | None = None, clock=time.monotonic) -> bool:
    """Block the calling worker thread until busy() is False, or until
    deadline_s has passed. Returns whether it became idle.

    Side effects: sleeps. Used before an update restart and Quit, because
    shutdown discards any recording or transcription still in progress.
    """
    give_up = None if deadline_s is None else clock() + deadline_s
    while busy():
        if give_up is not None and clock() >= give_up:
            return False
        sleep(poll_s)
    return True


def dictation_in_flight(capture, jobs, deliveries: PendingDeliveries, *, warming: bool = False) -> bool:
    """Whether a dictation is still recording, transcribing, or waiting to be
    pasted — the three stages Quit lets finish before shutting down.
    ``warming``: the one queued or running model rewarm is not a dictation (N40)."""
    return (capture.is_active() or jobs.unfinished_tasks > int(warming)
            or deliveries.count() > 0)


def end_capture_now(engine, capture, on_finish) -> str | None:
    """End the recording in progress so it is transcribed, not dropped.

    Returns "gesture" when the gesture engine ended it, "orphan" when the
    capture outlived the engine's state (a gesture callback raised after the
    mic started) and was ended directly, or None when nothing was recording.
    ``on_finish`` ends it either way (for a gesture the engine runs it, maybe
    later on its drain thread): the Mac passes the engine's own on_finish,
    Windows a finish that copies instead of inserting (a tray action took the
    focus).
    Side effects: may stop the capture and queue it for transcription.
    """
    if engine.force_finish(finish=on_finish):
        return "gesture"
    if capture.is_active():
        on_finish()
        return "orphan"
    return None


class CaptureGate:
    """Whether a key press may start a new recording.

    close() is called before an update restart: it waits for a start already
    under way (so that recording is visible as active), then refuses new ones,
    so "wait until idle, then restart" cannot discard a recording that began
    in between. reopen() undoes it if the restart does not happen.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._open = True

    @contextmanager
    def starting(self):
        """Hold while deciding and starting a recording; yields whether it may start."""
        with self._lock:
            yield self._open

    def close(self) -> None:
        with self._lock:
            self._open = False

    def reopen(self) -> None:
        with self._lock:
            self._open = True


class Lifecycle:
    """Who holds the capture gate closed, and the drained stop.

    Quit, the menu Restart, an installed update and an engine switch close
    the gate for a reason; ``closed`` names it ("quitting", "restarting",
    "updating", "switching engines") and is None while Scrivo runs normally.
    A reason reopens the gate only while it still holds it; Quit takes over
    from any other reason.

    Quit, the menu Restart and the first SIGINT/SIGTERM stop the same way
    (F1c, N12): refuse new recordings, end the one in progress as a normal
    finish, wait until it is transcribed and pasted — at most QUIT_DRAIN_S,
    SIGTERM_DRAIN_S for SIGTERM — then shut down or restart. What is still
    untranscribed past that wait is kept in History at teardown (N9).
    """

    def __init__(self, shutdown: ShutdownBoundary, capture_gate: CaptureGate, *,
                 end_recording, busy, clock=time.monotonic, sleep=time.sleep) -> None:
        self.shutdown = shutdown
        self.capture_gate = capture_gate
        self.end_recording = end_recording  # -> "gesture" | "orphan" | None (end_capture_now)
        self.busy = busy                    # -> whether a dictation is in flight
        self.clock, self.sleep = clock, sleep
        self.closed: str | None = None
        self._quit_by = 0.0
        self._lock = threading.Lock()

    def close(self, reason: str) -> bool:
        """Refuse new recordings for ``reason``. False (and nothing changes)
        when Scrivo is stopping or the gate is already closed for a reason."""
        with self._lock:
            if self.shutdown.requested() or self.closed is not None:
                return False
            self.closed = reason
            self.capture_gate.close()
            return True

    def reopen(self, reason: str) -> None:
        """Undo close(reason), unless another reason (a Quit) has taken over."""
        with self._lock:
            if self.closed == reason and not self.shutdown.requested():
                self.closed = None
                self.capture_gate.reopen()

    def run_if_open(self, action) -> str | None:
        """Run ``action`` unless the gate is closed; returns why it was refused.
        Checked and run under the lock, so a drain that closes the gate
        afterwards sees whatever ``action`` queued as in flight."""
        with self._lock:
            if self.shutdown.requested():
                return "stopping"
            if self.closed is not None:
                return self.closed
            action()
            return None

    def quit(self, deadline_s: float = QUIT_DRAIN_S) -> bool:
        """Stop once the dictation in flight is done, waiting at most
        ``deadline_s``. Returns at once; False when already quitting.

        Side effects: closes the gate as "quitting"; ends the recording in
        progress; requests shutdown from a thread once idle or out of time."""
        with self._lock:
            if self.shutdown.requested():
                return False
            stop_by = self.clock() + deadline_s
            if self.closed == "quitting":
                # Already draining: a SIGTERM after the menu's Quit must still
                # stop inside launchd's window, so the sooner deadline wins.
                self._quit_by = min(self._quit_by, stop_by)
                return False
            self.closed = "quitting"
            self._quit_by = stop_by
            self.capture_gate.close()
        self._finish_recording("quitting")
        self._drain_then("quitting", lambda: self._quit_by, self.shutdown.request)
        return True

    def restart(self, request_restart) -> bool:
        """The menu Restart: drain like Quit, then call
        ``request_restart(after_failure=...)``, which reopens the gate if the
        restart does not happen. False when the gate is already closed."""
        if not self.close("restarting"):
            return False
        self._finish_recording("restarting")

        def restart_unless_quitting() -> None:
            if self.closed == "restarting":
                request_restart(after_failure=lambda: self.reopen("restarting"))
        stop_by = self.clock() + QUIT_DRAIN_S
        self._drain_then("restarting", lambda: stop_by, restart_unless_quitting)
        return True

    def watch_signals(self) -> None:
        """Thread target: the first SIGINT/SIGTERM quits as the menu's Quit
        does (the handler itself only sets an event)."""
        self.shutdown.stop_event.wait()
        self.quit(SIGTERM_DRAIN_S if self.shutdown.stop_signal == signal.SIGTERM else QUIT_DRAIN_S)

    def _finish_recording(self, verb: str) -> None:
        try:
            ended = self.end_recording()
        except Exception as exc:
            ended = None
            log(f"! could not finish the recording before {verb}: {str(exc)[:160]}")
        if ended:
            log(f"● finishing the {'orphan capture' if ended == 'orphan' else 'recording'} before {verb}")

    def _drain_then(self, verb: str, stop_by, then) -> None:
        """From a thread: wait while busy, until the clock reaches ``stop_by()``
        (re-read every poll), then call ``then``."""
        def work() -> None:
            started = self.clock()
            try:
                while self.busy() and not self.shutdown.requested():
                    if self.clock() >= stop_by():
                        log(f"! dictation still running after {self.clock() - started:.0f}s — "
                            f"{verb} anyway; its audio stays in History")
                        break
                    self.sleep(DRAIN_POLL_S)
            except Exception as exc:
                log(f"! {verb} without waiting: {str(exc)[:160]}")
            finally:
                then()
        threading.Thread(target=work, daemon=True, name=f"sotto-{verb}").start()
