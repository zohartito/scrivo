"""The trigger key, the remote-friendly hotkey and the event tap that hears them.

TRIGGERS/TRIGGER_LABELS name the bare-modifier trigger keys; parse_hotkey
reads the alternate hotkey ("ctrl-opt-d"). HotkeyListener owns the session
event tap: its callback turns trigger edges, chords and hotkey presses into
GestureEngine calls, and its resync poller revives a tap macOS disabled
silently and synthesizes a release the tap missed.

Moved out of sotto.py (restructure step 6): the tables and helpers verbatim,
the tap closures of run() as HotkeyListener methods. sotto re-exports the
public names, so ``sotto.TRIGGERS is hotkey.TRIGGERS``. Quartz is never
imported here: run() passes the real module as the ``quartz`` adapter, tests
pass a fake, so the tap's decisions are testable without macOS. Imports
nothing from sotto.
"""
from __future__ import annotations

import sys
import threading

from gestures import HOLD_THRESHOLD_S

# CGEventFlags masks (CGEventTypes.h), as literals so this module imports
# without PyObjC (Windows, Linux). tests/test_hotkey.py pins each one to
# Quartz's own value on macOS.
kCGEventFlagMaskShift = 0x00020000
kCGEventFlagMaskControl = 0x00040000
kCGEventFlagMaskAlternate = 0x00080000
kCGEventFlagMaskCommand = 0x00100000
kCGEventFlagMaskSecondaryFn = 0x00800000

# trigger name -> (modifier flag mask, virtual keycode, per-side device bit).
# The NX_DEVICE*KEYMASK bits identify WHICH side of a paired modifier is down —
# the aggregate mask alone lies when the other side's twin is held. (Verified
# live on this machine: right-option down -> flags contain 0x40, not 0x20.)
# CGEventSourceKeyState does NOT track modifier keycodes; never use it here.
TRIGGERS = {
    "fn": (kCGEventFlagMaskSecondaryFn, 63, 0),
    "right-cmd": (kCGEventFlagMaskCommand, 54, 0x0010),
    "right-option": (kCGEventFlagMaskAlternate, 61, 0x0040),
    "right-shift": (kCGEventFlagMaskShift, 60, 0x0004),
    "right-ctrl": (kCGEventFlagMaskControl, 62, 0x2000),
    "left-cmd": (kCGEventFlagMaskCommand, 55, 0x0008),
    "left-option": (kCGEventFlagMaskAlternate, 58, 0x0020),
    "left-shift": (kCGEventFlagMaskShift, 56, 0x0002),
    "left-ctrl": (kCGEventFlagMaskControl, 59, 0x0001),
}
TRIGGER_LABELS = {
    "right-option": "Right Option ⌥", "left-option": "Left Option ⌥",
    "right-cmd": "Right Command ⌘", "left-cmd": "Left Command ⌘",
    "right-ctrl": "Right Control ⌃", "left-ctrl": "Left Control ⌃",
    "right-shift": "Right Shift ⇧", "left-shift": "Left Shift ⇧", "fn": "Fn / Globe",
}


def hands_free_hint(trigger: str) -> str:
    """How to end hands-free, short enough for the overlay: "tap right ⌘ to stop"."""
    if trigger == "fn":
        return "tap fn to stop"
    words = TRIGGER_LABELS.get(trigger, trigger).split()
    return f"tap {words[0].lower()} {words[-1]} to stop"


# Toggle hotkey: a NORMAL key plus modifiers. Unlike the bare-modifier
# trigger, real keycodes survive remote-desktop sessions (AnyDesk from an
# iPad), and toggle semantics don't depend on release timing over a network.
KEYCODES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8,
    "v": 9, "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "o": 31, "u": 32, "i": 34, "p": 35, "l": 37, "j": 38, "k": 40, "n": 45,
    "m": 46, "space": 49, "f13": 105, "f14": 107, "f15": 113, "f16": 106,
}
MODIFIER_FLAGS = {
    "cmd": kCGEventFlagMaskCommand,
    "ctrl": kCGEventFlagMaskControl,
    "opt": kCGEventFlagMaskAlternate,
    "alt": kCGEventFlagMaskAlternate,
    "shift": kCGEventFlagMaskShift,
}
ALL_MODIFIERS = (kCGEventFlagMaskCommand | kCGEventFlagMaskControl
                 | kCGEventFlagMaskAlternate
                 | kCGEventFlagMaskShift)


def parse_hotkey(spec: str) -> tuple[int, int] | None:
    """'ctrl-opt-d' -> (required modifier flags, keycode). None if unparsable."""
    parts = [p.strip().lower() for p in spec.split("-") if p.strip()]
    if not parts:
        return None
    key = parts[-1]
    if key not in KEYCODES:
        return None
    flags = 0
    for part in parts[:-1]:
        if part not in MODIFIER_FLAGS:
            return None
        flags |= MODIFIER_FLAGS[part]
    return flags, KEYCODES[key]


def trigger_down_in(flags: int, mask: int, device_bit: int) -> bool:
    """Is the trigger key down according to these event/source flags?"""
    if not flags & mask:
        return False
    return bool(flags & device_bit) if device_bit else True


def hotkey_release_held_seconds(state: dict, now: float) -> float | None:
    """Consume a hotkey key-up only after an accepted matching key-down.

    Key-up events no longer carry reliable modifier flags, so the event tap
    matches them by keycode alone.  That makes the stored key-down the only
    authority: without it, a plain/stray D key-up must not finish some other
    trigger's active recording.  ``None`` also covers a release intentionally
    consumed after a toggle press already finished the recording.
    """
    pressed_at = state.get("pressed_at")
    if pressed_at is None:
        state["skip_up"] = False
        return None
    state["pressed_at"] = None
    if state.get("skip_up"):
        state["skip_up"] = False
        return None
    state["skip_up"] = False
    return max(0.0, now - float(pressed_at))


class HotkeyListener:
    """The session event tap: trigger edges and the hotkey drive ``engine``.

    Collaborators are injected: ``quartz`` is the Quartz module (or a fake
    with the same names), ``capture`` is ticked by the poller,
    ``user_keydowns`` is the {"count": n} DeliveryQueue reads, ``ui_call``/
    ``status_ui`` show tap health, ``clock``/``sleep`` are time.monotonic and
    time.sleep. ``binding`` is shared with the Settings window (AppController)
    and ``hotkey`` is the parsed hotkey or None; ``rebind`` changes both.
    """

    def __init__(self, trigger: str, hotkey_spec: str, *, quartz, engine, capture, shutdown, is_own_event,
                 user_keydowns: dict, log, status_ui, ui_call, clock, sleep):
        self.quartz = quartz
        self.engine = engine
        self.capture = capture
        self.shutdown = shutdown
        self.is_own_event = is_own_event
        self.user_keydowns = user_keydowns
        self.log = log
        self.status_ui = status_ui
        self.ui_call = ui_call
        self.clock = clock
        self.sleep = sleep
        self.tap = None
        self.hotkey = parse_hotkey(hotkey_spec) if hotkey_spec else None
        self.hotkey_state: dict = {"pressed_at": None, "skip_up": False}
        if hotkey_spec and self.hotkey is None:
            self.log(f"! unrecognized --hotkey '{hotkey_spec}' — hotkey disabled")
        # One mutable binding so Settings can move the trigger at the next press.
        self.binding = {"trigger": trigger, "hotkey_spec": hotkey_spec if self.hotkey else ""}
        self.binding["mask"], self.binding["keycode"], self.binding["device_bit"] = TRIGGERS[trigger]
        self.trigger_down = False
        self.tap_warned_at = 0.0

    def trigger_physically_down(self) -> bool:
        return trigger_down_in(
            self.quartz.CGEventSourceFlagsState(
                self.quartz.kCGEventSourceStateCombinedSessionState),
            self.binding["mask"], self.binding["device_bit"])

    def rebind(self, new_trigger: str | None = None, new_hotkey_spec: str | None = None) -> None:
        """Apply a Settings change to the live listener; drops any held capture."""
        if new_trigger is not None:
            self.binding["mask"], self.binding["keycode"], self.binding["device_bit"] = TRIGGERS[new_trigger]
            self.binding["trigger"] = new_trigger
            self.trigger_down = False
            self.engine.force_reset()
            self.log(f"● trigger is now {new_trigger}")
        if new_hotkey_spec is not None:
            parsed = parse_hotkey(new_hotkey_spec) if new_hotkey_spec else None
            self.hotkey = parsed
            self.binding["hotkey_spec"] = new_hotkey_spec if parsed else ""
            self.hotkey_state.update(pressed_at=None, skip_up=False)
            self.log(f"● alternate hotkey {'is now ' + new_hotkey_spec if parsed else 'off'}")

    def keys_held(self) -> bool:
        """Is the trigger or any modifier physically down right now?"""
        flags = self.quartz.CGEventSourceFlagsState(self.quartz.kCGEventSourceStateCombinedSessionState)
        return bool(flags & ALL_MODIFIERS) or trigger_down_in(
            flags, self.binding["mask"], self.binding["device_bit"])

    def revive_tap(self, reason: str) -> None:
        """Re-arm the listener and resync the trigger edge it may have missed.

        macOS disables a session tap on callback timeout (with a notification)
        but also silently — Accessibility churn, display/session changes — and
        a deaf tap looks exactly like a working one from the outside (2026-09-15:
        3 hours of dead hotkey, process healthy, tap enabled=False)."""
        self.quartz.CGEventTapEnable(self.tap, True)
        current = self.trigger_physically_down()
        if self.trigger_down and not current:
            self.trigger_down = False
            self.engine.released()  # the release edge happened while deaf
        else:
            self.trigger_down = current
        if self.quartz.CGEventTapIsEnabled(self.tap):
            self.log(f"! event tap re-enabled after {reason}")
            self.tap_warned_at = 0.0
            if self.status_ui is not None:
                self.ui_call(self.status_ui.set_tap_health, True)
        elif self.clock() - self.tap_warned_at > 60:
            self.tap_warned_at = self.clock()
            self.log("✗ event tap stays disabled — re-grant Accessibility to Sotto")
            if self.status_ui is not None:
                self.ui_call(self.status_ui.set_tap_health, False)

    def callback(self, _proxy, event_type, event, _refcon):
        if self.shutdown.requested():
            return event
        if event_type in (self.quartz.kCGEventKeyDown, self.quartz.kCGEventKeyUp,
                          self.quartz.kCGEventFlagsChanged) and self.is_own_event(event):
            return event  # Sotto's own ⌘V / ⌘Z / typing: never a chord, never the user typing
        if event_type == self.quartz.kCGEventKeyDown:
            code = self.quartz.CGEventGetIntegerValueField(event, self.quartz.kCGKeyboardEventKeycode)
            hotkey_key = self.hotkey is not None and code == self.hotkey[1]
            if not (hotkey_key and (self.quartz.CGEventGetFlags(event) & ALL_MODIFIERS) == self.hotkey[0]):
                self.user_keydowns["count"] += 1  # the user typed: "scratch that" is off the table
            if self.trigger_down and not hotkey_key:
                self.engine.chorded()  # any other key during a trigger hold makes it a shortcut
        # Hotkey: hold it and release to stop, or tap it and tap again later.
        # Normal keys carry real keycodes AND key-up events through remote
        # desktops, so both gestures survive an AnyDesk session.
        if self.hotkey is not None and event_type in (self.quartz.kCGEventKeyDown,
                                                      self.quartz.kCGEventKeyUp):
            want_flags, want_code = self.hotkey
            code = self.quartz.CGEventGetIntegerValueField(
                event, self.quartz.kCGKeyboardEventKeycode)
            if code != want_code:
                return event
            if event_type == self.quartz.kCGEventKeyDown:
                if self.quartz.CGEventGetIntegerValueField(
                        event, self.quartz.kCGKeyboardEventAutorepeat):
                    return event  # holding the key, not a new press
                if (self.quartz.CGEventGetFlags(event) & ALL_MODIFIERS) != want_flags:
                    return event
                if self.trigger_down:
                    self.engine.chorded()  # the trigger was part of the hotkey chord
                if self.engine.snapshot()[0]:
                    # a tap-started session is running — this press ends it
                    self.hotkey_state["pressed_at"] = self.clock()
                    self.hotkey_state["skip_up"] = True
                    if self.engine.force_finish():
                        self.log("● finished via hotkey")
                elif self.engine.force_start():
                    self.hotkey_state["skip_up"] = False
                    self.hotkey_state["pressed_at"] = self.clock()
                    self.log("● dictation started via hotkey")
            else:  # key up — flags may already be gone, match on keycode alone
                held = hotkey_release_held_seconds(
                    self.hotkey_state, self.clock())
                if (held is not None and held >= HOLD_THRESHOLD_S
                        and self.engine.snapshot()[0]):
                    if self.engine.force_finish():
                        self.log(f"● finished (held {held:.1f}s)")
                # a short tap leaves the session running until the next press
            return event
        if event_type in (self.quartz.kCGEventTapDisabledByTimeout,
                          self.quartz.kCGEventTapDisabledByUserInput):
            self.revive_tap("system disable")  # never die silently
            return event
        if event_type == self.quartz.kCGEventFlagsChanged:
            if self.quartz.CGEventGetIntegerValueField(
                    event, self.quartz.kCGKeyboardEventKeycode) == self.binding["keycode"]:
                down = trigger_down_in(self.quartz.CGEventGetFlags(event),
                                       self.binding["mask"], self.binding["device_bit"])
                if down != self.trigger_down:
                    self.trigger_down = down
                    (self.engine.pressed if down else self.engine.released)()
            elif self.trigger_down:
                self.engine.chorded()  # another modifier joined the hold
        return event

    def resync_poller(self) -> None:
        """Escape hatch: if a release edge is ever lost (tap deafness, phantom
        edges, sleep), a push-to-talk recording whose key is physically up for
        two consecutive polls gets its release synthesized. Hands-free is
        exempt — its key is legitimately up while recording."""
        misses = 0
        while not self.shutdown.requested():
            self.sleep(1.0)
            if self.shutdown.requested():
                break
            self.capture.tick()
            if not self.quartz.CGEventTapIsEnabled(self.tap):
                # Directly on this thread — NEVER via callAfter. Bouncing to
                # the main loop was why the 2026-09-15 watchdog never fired
                # (2026-09-16: tap dead again, zero revive log lines) and
                # CGEventTapEnable is safe from a background thread.
                self.revive_tap("silent disable")
            recording, hands_free = self.engine.snapshot()
            if recording and not hands_free and not self.trigger_physically_down():
                misses += 1
                if misses >= 2:
                    self.log("! lost release recovered — finishing recording")
                    self.trigger_down = False
                    self.engine.released()
                    misses = 0
            else:
                misses = 0

    def start(self) -> None:
        """Create the listen-only session tap on the current run loop, enable
        it and start the resync poller; exit 1 when macOS refuses the tap.

        Side effects: an event tap, a run-loop source, a daemon thread; logs."""
        self.tap = self.quartz.CGEventTapCreate(
            self.quartz.kCGSessionEventTap, self.quartz.kCGHeadInsertEventTap,
            self.quartz.kCGEventTapOptionListenOnly,
            self.quartz.CGEventMaskBit(self.quartz.kCGEventFlagsChanged)
            | self.quartz.CGEventMaskBit(self.quartz.kCGEventKeyDown)
            | self.quartz.CGEventMaskBit(self.quartz.kCGEventKeyUp), self.callback, None,
        )
        if self.tap is None:
            self.log("failed to create event tap — grant Accessibility to this terminal "
                     "(System Settings → Privacy & Security → Accessibility)")
            sys.exit(1)
        source = self.quartz.CFMachPortCreateRunLoopSource(None, self.tap, 0)
        self.quartz.CFRunLoopAddSource(self.quartz.CFRunLoopGetCurrent(), source,
                                       self.quartz.kCFRunLoopCommonModes)
        self.quartz.CGEventTapEnable(self.tap, True)
        threading.Thread(target=self.resync_poller, daemon=True).start()
