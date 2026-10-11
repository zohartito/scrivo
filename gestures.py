"""Trigger-key gestures: press/release edges become start, finish and discard.

hold = push-to-talk (release finishes), double-tap = hands-free (the next tap
finishes; a watchdog finishes a forgotten one after HANDS_FREE_MAX_S), a lone
tap is discarded after DOUBLE_TAP_WINDOW_S. Shared by the Mac event tap and
the Windows keyboard hook.

Moved verbatim out of sotto.py (restructure step 3); sotto re-exports the
public names, so ``sotto.GestureEngine is gestures.GestureEngine``. The engine
reads its timing constants from this module: patch or override them here
(``gestures.HOLD_THRESHOLD_S``), never through sotto. Imports nothing from
sotto.
"""
from __future__ import annotations

import threading
import time

from sotto_log import log


HOLD_THRESHOLD_S = 0.35
DOUBLE_TAP_WINDOW_S = 0.40
HANDS_FREE_MAX_S = 600.0        # watchdog: force-finish a forgotten open mic


class GestureEngine:
    """Press/release edges -> hold / double-tap / discard decisions.

    Every key deadline carries an epoch token: any newer key event invalidates
    it, so a Timer callback that already fired but is waiting on the lock can
    never act on stale state. The hands-free watchdog is the exception: it
    belongs to the capture, not the key epoch, so a stop tap whose release is
    lost cannot disarm it — only ending or re-arming hands-free does.
    Callbacks run OUTSIDE the lock, one at a time, in the order their
    decisions were made: a decision queues its actions while it still holds
    the lock, and whichever thread finds nobody draining runs the queue (see
    _drain). The second press of a double-tap is classified on its RELEASE:
    held short = arm hands-free, held long = it was a deliberate push-to-talk.
    """

    def __init__(self, on_start, on_finish, on_discard, on_hands_free=None) -> None:
        self._on_start = on_start
        self._on_finish = on_finish
        self._on_discard = on_discard
        self._on_hands_free = on_hands_free
        self._lock = threading.Lock()
        self._epoch = 0
        self._hands_free_token = 0
        self._recording = False
        self._hands_free = False
        self._tap_pending = False      # a lone short tap awaits its verdict
        self._second_candidate = False
        self._pressed_at = 0.0
        self._last_tap_at = -1e9
        self._pending: list = []       # actions decided, not yet run (in order)
        self._draining = False

    def snapshot(self) -> tuple[bool, bool]:
        """(recording, hands_free) — for the lost-release resync poller."""
        with self._lock:
            return self._recording, self._hands_free

    def _queue(self, callbacks) -> None:
        """Caller holds the lock: actions join the queue in decision order."""
        self._pending.extend(callbacks)

    def _drain(self) -> None:
        """Run queued actions one at a time, in order, outside the lock.

        The thread that finds nobody draining becomes the drainer and runs
        everything queued, including what other threads add meanwhile; a
        thread that arrives while another is draining has already queued its
        actions under the lock and returns at once. So the event tap never
        waits on a timer thread's callback, and a discard decided before a
        start (the tap-expiry timer racing the next press) can never run
        after it and kill the new capture."""
        with self._lock:
            if self._draining:
                return
            self._draining = True
        while True:
            with self._lock:
                if not self._pending:
                    self._draining = False
                    return
                callback = self._pending.pop(0)
            try:
                callback()
            except Exception as exc:
                log(f"! gesture callback failed: {str(exc)[:120]}")
                with self._lock:
                    self._recording = False
                    self._hands_free = False

    def pressed(self) -> None:
        fires = []
        with self._lock:
            self._epoch += 1
            now = time.monotonic()
            candidate = self._tap_pending and now - self._last_tap_at < DOUBLE_TAP_WINDOW_S
            stale_blip = self._tap_pending and not candidate
            self._tap_pending = False
            self._second_candidate = candidate and self._recording and not self._hands_free
            if stale_blip and self._recording and not self._hands_free:
                # the discard timer ran late — drop the old blip, start fresh
                self._recording = False
                fires.append(self._on_discard)
            if not self._recording:
                self._recording = True
                fires.append(self._on_start)
            self._pressed_at = now
            self._queue(fires)
        self._drain()

    def released(self) -> None:
        fires = []
        with self._lock:
            now = time.monotonic()
            held_for = now - self._pressed_at
            if not self._recording:
                pass
            elif self._hands_free:
                self._hands_free = False
                self._recording = False
                fires.append(self._on_finish)
            elif held_for >= HOLD_THRESHOLD_S:
                self._recording = False
                fires.append(self._on_finish)
            elif self._second_candidate:
                self._hands_free = True
                self._epoch += 1
                self._arm_hands_free_watchdog()
                log("● hands-free (tap again to stop)")
                if self._on_hands_free is not None:
                    fires.append(self._on_hands_free)
            else:
                self._last_tap_at = now
                self._tap_pending = True
                self._epoch += 1
                self._schedule(DOUBLE_TAP_WINDOW_S, self._expire_tap)
            self._second_candidate = False
            self._queue(fires)
        self._drain()

    def force_start(self) -> bool:
        """Menu-driven start: arm a hands-free recording without any key
        gesture (remote desktops send synthetic keys without the physical
        device bits, so the hotkey can't work through them)."""
        fires = []
        with self._lock:
            if self._recording:
                return False
            self._epoch += 1
            self._tap_pending = False
            self._recording = True
            self._hands_free = True
            self._arm_hands_free_watchdog()
            fires.append(self._on_start)
            if self._on_hands_free is not None:
                fires.append(self._on_hands_free)
            self._queue(fires)
        self._drain()
        return True

    def epoch(self) -> int:
        """The key-decision counter: it moves on every press, release and forced end."""
        with self._lock:
            return self._epoch

    def force_finish(self, finish=None, if_epoch: int | None = None) -> bool:
        """Menu escape hatch: end any in-flight recording as a normal finish.

        `finish` replaces on_finish for this one decision. It may run after
        this returns (another thread is draining), so anything the caller
        wants that finish to know must travel inside it — never in state the
        caller sets around this call. `if_epoch` ties the finish to the
        gesture current at that epoch(): after any later key decision nothing
        changes and False is returned, so a late failure of an older capture
        can never end a newer gesture."""
        fires = []
        with self._lock:
            if if_epoch is not None and self._epoch != if_epoch:
                return False
            self._epoch += 1
            self._tap_pending = False
            was_recording = self._recording
            if self._recording:
                self._recording = False
                self._hands_free = False
                fires.append(finish or self._on_finish)
            self._queue(fires)
        self._drain()
        return was_recording

    def chorded(self) -> None:
        """Another key joined a held trigger (⌘-Tab, ⌥-letter): that is a
        shortcut, not dictation. Drop the push-to-talk capture; hands-free
        keeps recording because its trigger is no longer held."""
        fires = []
        with self._lock:
            if not self._recording or self._hands_free:
                return
            self._epoch += 1
            self._tap_pending = False
            self._second_candidate = False
            self._recording = False
            fires.append(self._on_discard)
            self._queue(fires)
        self._drain()

    def force_reset(self) -> None:
        """Sleep/lock/tap-disable recovery: drop any in-flight recording."""
        fires = []
        with self._lock:
            self._epoch += 1
            self._tap_pending = False
            self._second_candidate = False
            if self._recording:
                self._recording = False
                self._hands_free = False
                fires.append(self._on_discard)
            self._queue(fires)
        self._drain()

    def _schedule(self, delay: float, handler, token: int | None = None) -> None:
        timer = threading.Timer(delay, handler,
                                args=(self._epoch if token is None else token,))
        timer.daemon = True
        timer.start()

    def _arm_hands_free_watchdog(self) -> None:
        """Caller holds the lock. The watchdog is bound to THIS hands-free
        capture: pressing the key again must not disarm it (a stop tap whose
        release is lost used to leave the mic open forever), and a re-armed
        hands-free gets a fresh token so the old deadline cannot end it."""
        self._hands_free_token += 1
        self._schedule(HANDS_FREE_MAX_S, self._hands_free_timeout,
                       self._hands_free_token)

    def _expire_tap(self, epoch: int) -> None:
        fires = []
        with self._lock:
            if epoch != self._epoch or not self._recording or self._hands_free:
                return
            self._recording = False
            self._tap_pending = False
            fires.append(self._on_discard)
            self._queue(fires)
        self._drain()

    def _hands_free_timeout(self, token: int) -> None:
        fires = []
        with self._lock:
            if token != self._hands_free_token or not self._hands_free:
                return
            self._hands_free = False
            self._recording = False
            log(f"! hands-free watchdog ({HANDS_FREE_MAX_S:.0f}s) — finishing")
            fires.append(self._on_finish)
            self._queue(fires)
        self._drain()
