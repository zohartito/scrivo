"""Handing finished text to the frontmost app.

inject() pastes through the general pasteboard (full snapshot, synthetic
Cmd-V, changeCount-guarded restore, carried forward across overlapping
pastes) or types it; every key event Sotto posts is tagged as its own.
DeliveryQueue is the one FIFO for pastes and "scratch that" undos;
PendingDeliveries counts what was handed to the main thread but not yet
delivered. Main thread only, except where a docstring says otherwise.

Moved verbatim out of sotto.py (restructure step 1); sotto re-exports the
public names, so ``sotto.inject is delivery.inject``. The clipboard restore
state (``_pending_restore``, ``_restore_generation``) and the secure-input
probe are module state rebound here, so they are read and patched here, never
through sotto. Imports nothing from sotto.
"""
from __future__ import annotations

import threading
import time

from sotto_log import log

try:  # Keep content-free CLI/admin helpers importable on non-macOS test hosts.
    import Quartz
    from AppKit import NSPasteboard, NSPasteboardItem, NSPasteboardTypeString
except ImportError:  # pragma: no cover - capture itself remains macOS-only
    class _UnavailableMacAPI:
        # Constants are declared at module load; capture methods fail only if
        # somebody actually starts the macOS-only runtime.
        def __getattr__(self, name): return 0
    Quartz = _UnavailableMacAPI()
    NSPasteboard = NSPasteboardItem = NSPasteboardTypeString = _UnavailableMacAPI()


# Clipboard restore after the synthetic ⌘V. Reading the pasteboard does not
# change its changeCount and there is no public "the app has read it" signal,
# so the restore is a bounded wait: long enough for an app that is briefly busy
# at paste time (Electron hitches, a tab mid-layout) to still read Sotto's text
# — the old 0.6 s handed such apps the user's OLD clipboard — and short enough
# that a user ⌘V a few seconds later gets their own clipboard back. The restore
# still runs only if changeCount is unchanged (a user ⌘C in the window wins),
# and overlapping pastes carry the original forward (see inject()).
RESTORE_DELAY_S = 3.0
# nspasteboard.org marker on Sotto's own paste write: clipboard managers skip
# it, so dictations do not pile up in their history (the restore is untouched).
TRANSIENT_PASTEBOARD_TYPE = "org.nspasteboard.TransientType"
DELIVERY_POLL_S = 0.15           # re-check a held key this often before pasting
DELIVERY_WAIT_MAX_S = 30.0       # key held, no recording: give up, the text stays in History
# The drop note for a dictation History could not keep (F16a/F16b, N16).
NOT_IN_HISTORY_EITHER = "History could not save it either; please dictate again."
SOTTO_EVENT_TAG = 0x534F5454     # "SOTT" in kCGEventSourceUserData on every key event Sotto posts


_restore_generation = 0
_pending_restore: dict | None = None  # {"own_count", "snapshot", "due"} of the restore not yet run


_NO_SPACE_AFTER = "([{\"'“‘/-\n\t"
_NO_SPACE_BEFORE = ".,;:!?)]}\"'”’%"


def compose_insertion(text: str, spacing: str, before: str | None) -> str:
    """The exact text to insert for a dictation.

    ``before`` is the character left of the caret ("" at the start of a
    field, None when the app does not expose it). "smart" adds a leading space
    only when the text would otherwise touch the previous word, and falls back
    to "trailing" when the context is unknown; "trailing" keeps consecutive
    dictations apart ('Hey what's upfirstI'm bursting'); "none" inserts as-is."""
    if not text or spacing == "none":
        return text
    if spacing == "smart" and before is not None:
        touching = bool(before) and not before.isspace() and before not in _NO_SPACE_AFTER
        return (" " + text) if touching and text[0] not in _NO_SPACE_BEFORE else text
    return text if text[-1].isspace() else text + " "


def character_before_caret(timeout: float = 0.15) -> str | None:
    """The character before the caret via Accessibility, or None when the
    focused app does not expose it (never blocks longer than ``timeout``)."""
    try:
        import ApplicationServices as AS
        system = AS.AXUIElementCreateSystemWide()
        AS.AXUIElementSetMessagingTimeout(system, timeout)
        error, focused = AS.AXUIElementCopyAttributeValue(system, AS.kAXFocusedUIElementAttribute, None)
        if error or focused is None:
            return None
        AS.AXUIElementSetMessagingTimeout(focused, timeout)
        error, selection = AS.AXUIElementCopyAttributeValue(focused, AS.kAXSelectedTextRangeAttribute, None)
        if error or selection is None:
            return None
        ok, (location, _length) = AS.AXValueGetValue(selection, AS.kAXValueCFRangeType, None)
        if not ok or location < 0:
            return None
        if location == 0:
            return ""
        window = AS.AXValueCreate(AS.kAXValueCFRangeType, AS.CFRange(location - 1, 1))
        error, previous = AS.AXUIElementCopyParameterizedAttributeValue(
            focused, AS.kAXStringForRangeParameterizedAttribute, window, None)
        if error or previous is None:
            return None
        return str(previous)[-1:] or None
    except Exception:
        return None


def post_own_event(event) -> None:
    """Tag a synthetic key event as Sotto's own and post it into the session.

    The event tap in run() skips tagged events, so a ⌘V or ⌘Z Sotto posts can
    never be read as a chord that discards a live dictation, nor as the user
    typing (which would disarm "scratch that")."""
    Quartz.CGEventSetIntegerValueField(event, Quartz.kCGEventSourceUserData, SOTTO_EVENT_TAG)
    Quartz.CGEventPost(Quartz.kCGSessionEventTap, event)


def is_own_event(event) -> bool:
    """Did Sotto post this key event itself (see post_own_event)?"""
    return Quartz.CGEventGetIntegerValueField(event, Quartz.kCGEventSourceUserData) == SOTTO_EVENT_TAG


def post_command_key(keycode: int) -> None:
    """Press and release ⌘+key (9 = V to paste, 6 = Z to undo) as Sotto's own events."""
    for key_down in (True, False):
        event = Quartz.CGEventCreateKeyboardEvent(None, keycode, key_down)
        Quartz.CGEventSetFlags(event, Quartz.kCGEventFlagMaskCommand)
        post_own_event(event)


def type_text(text: str) -> None:
    """Type Unicode text with synthetic key events; the clipboard is untouched."""
    characters = list(text)
    for start in range(0, len(characters), 16):  # small chunks: apps drop long ones
        chunk = "".join(characters[start:start + 16])
        units = len(chunk.encode("utf-16-le")) // 2
        for key_down in (True, False):
            event = Quartz.CGEventCreateKeyboardEvent(None, 0, key_down)
            Quartz.CGEventKeyboardSetUnicodeString(event, units, chunk)
            post_own_event(event)
        time.sleep(0.004)


_secure_input_probe = None


def secure_input_active() -> bool:
    """Is a password field (or another secure-input owner) active right now?

    PyObjC does not bridge Carbon's IsSecureEventInputEnabled, so this calls
    it directly; an unavailable probe reports False rather than blocking."""
    global _secure_input_probe
    try:
        if _secure_input_probe is None:
            import ctypes
            carbon = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Carbon.framework/Carbon")
            carbon.IsSecureEventInputEnabled.restype = ctypes.c_bool
            _secure_input_probe = carbon.IsSecureEventInputEnabled
        return bool(_secure_input_probe())
    except Exception:
        return False


def frontmost_pid() -> int | None:
    """The process a synthetic key event would reach right now (NSWorkspace)."""
    try:
        import AppKit
        app = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
        return int(app.processIdentifier()) if app is not None else None
    except Exception:
        return None


def own_window_without_text_field() -> bool:
    """Is Sotto itself the active app with no text field to take a paste?

    Sotto's alerts activate it (ui._alert), and callAfter still runs during
    their modal loop, so a Cmd-V would land on an OK-only alert and vanish
    (N24). The correction editor's text view still takes dictation. Main
    thread only; an unavailable probe reports False rather than blocking."""
    try:
        import AppKit
        app = AppKit.NSApp
        if app is None or not app.isActive():
            return False
        window = app.keyWindow()
        responder = window.firstResponder() if window is not None else None
        return responder is None or not responder.isKindOfClass_(AppKit.NSText)
    except Exception:
        return False


def inject(text: str, *, insert_mode: str = "paste", spacing: str = "trailing") -> bool:
    """Insert at the cursor. "paste": full-pasteboard snapshot, synthetic
    cmd-V, then a changeCount-guarded restore so a user copy in the window
    always wins. "type": synthetic Unicode typing, clipboard untouched.

    Returns True when the text was handed to the app, False when secure input
    declined it — only a real insert may arm "scratch that"."""
    global _restore_generation, _pending_restore
    if secure_input_active():
        log("! secure input active — not inserting")
        return False

    text = compose_insertion(text, spacing, character_before_caret() if spacing == "smart" else None)
    if insert_mode == "type":
        type_text(text)
        return True

    pasteboard = NSPasteboard.generalPasteboard()
    pending = _pending_restore
    if pending is not None and pasteboard.changeCount() == pending["own_count"]:
        # The pasteboard still holds Sotto's previous dictation and its restore
        # has not run yet: carry the user's original forward instead of
        # snapshotting our own text (two pastes inside the restore window).
        snapshot = pending["snapshot"]
    else:
        snapshot = []
        for item in (pasteboard.pasteboardItems() or []):
            snapshot.append([(t, item.dataForType_(t)) for t in item.types()])
    pasteboard.clearContents()
    pasteboard.setString_forType_(text, NSPasteboardTypeString)
    pasteboard.setData_forType_(b"", TRANSIENT_PASTEBOARD_TYPE)
    own_count = pasteboard.changeCount()
    post_command_key(9)  # ⌘V

    _restore_generation += 1
    generation = _restore_generation
    _pending_restore = {"own_count": own_count, "snapshot": snapshot,
                        "due": time.monotonic() + RESTORE_DELAY_S}

    def queue_restore() -> None:
        from PyObjCTools import AppHelper
        AppHelper.callAfter(_restore_clipboard, generation, own_count, snapshot)

    timer = threading.Timer(RESTORE_DELAY_S, queue_restore)
    timer.daemon = True
    timer.start()
    return True


def _restore_clipboard(generation: int, own_count: int, snapshot: list) -> None:
    """Runs on the main thread (AppHelper.callAfter), like inject() itself, so
    the pending-restore bookkeeping needs no lock."""
    global _pending_restore
    if generation != _restore_generation:
        return  # a newer injection owns the pasteboard now (and carried this snapshot if it was still ours)
    _pending_restore = None
    pasteboard = NSPasteboard.generalPasteboard()
    if pasteboard.changeCount() != own_count:
        return  # the user copied something meanwhile — their copy wins
    pasteboard.clearContents()
    items = []
    for entry in snapshot:
        item = NSPasteboardItem.alloc().init()
        for pb_type, data in entry:
            if data is not None:
                item.setData_forType_(data, pb_type)
        items.append(item)
    if items:
        pasteboard.writeObjects_(items)


def flush_clipboard_restore(max_wait_s: float = RESTORE_DELAY_S, *, clock=time.monotonic,
                            sleep=time.sleep) -> None:
    """Run a clipboard restore that is still pending, before the process ends
    (N6): its timer would otherwise die with it, leaving the dictation on the
    clipboard and the user's own copy lost. Waits until the restore is due,
    at most ``max_wait_s``, so the app can still read the paste; the restore
    keeps its changeCount guard, so a user copy since the paste still wins.

    Side effects: may sleep; may rewrite the general pasteboard. Main thread
    only, like _restore_clipboard."""
    pending = _pending_restore
    if pending is None:
        return
    wait_s = min(max(pending["due"] - clock(), 0.0), max_wait_s)
    if wait_s > 0:
        sleep(wait_s)
    _restore_clipboard(_restore_generation, pending["own_count"], pending["snapshot"])


def set_clipboard_text(text: str) -> None:
    """Replace the general clipboard with ``text`` (main thread only).

    Side effects: rewrites the general pasteboard."""
    pasteboard = NSPasteboard.generalPasteboard()
    pasteboard.clearContents()
    pasteboard.setString_forType_(text, NSPasteboardTypeString)


class PendingDeliveries:
    """Pastes and undos handed to the main thread but not yet delivered.

    The transcription queue counts a job done once its paste is scheduled,
    before the main loop runs it, so Quit also waits for this to reach zero.
    add() runs on the worker, finish() on the main thread; hence the lock.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._count = 0

    def add(self) -> None:
        with self._lock:
            self._count += 1

    def finish(self) -> None:
        with self._lock:
            self._count = max(0, self._count - 1)

    def count(self) -> int:
        with self._lock:
            return self._count


class DeliveryQueue:
    """One FIFO for everything Sotto hands to the frontmost app — finished
    dictations and "scratch that" undos — in capture order, so a newer result
    can never overtake an older one.

    Nothing is posted while the trigger or any modifier is physically held: a
    synthetic ⌘V under a held modifier is a different shortcut in the app, and
    during a hold macOS emits phantom flag edges on the trigger keycode that
    chop the live dictation. The head of the queue waits as long as a
    recording is active (the release delivers it); with no recording, a key
    held longer than DELIVERY_WAIT_MAX_S drops the pending text — it is already
    in History — with one note. While Sotto's own alert is in front
    (own_window_front), the head waits without a bound: a Cmd-V there would
    land nowhere, and the user's next app switch or click on OK delivers it.

    Every method runs on the main thread (deliver_call / AppHelper.callAfter);
    the poll timer only bounces back there, so there is no lock.

    on_done runs exactly once for every item the queue takes — when it is
    delivered (even if the insert raises), dropped after a long hold, or
    cleared at shutdown — so run() can count it in PendingDeliveries from the
    moment deliver_call schedules it until it leaves the queue.
    """

    def __init__(self, *, insert, undo_keys, keys_held, recording, frontmost_pid, keydowns,
                 secure_input, call_after, note, log=log, shutdown_requested=lambda: False,
                 clock=time.monotonic, timer=threading.Timer, on_done=lambda: None,
                 own_window_front=lambda: False, copy=lambda text: None) -> None:
        self._insert = insert            # (text) -> bool: did the text reach the app?
        self._undo_keys = undo_keys      # () -> None: press the app's own ⌘Z
        self._keys_held = keys_held      # () -> bool: trigger or modifier physically down
        self._recording = recording      # () -> bool: a dictation is being captured
        self._frontmost_pid = frontmost_pid  # () -> pid of the app a key event reaches
        self._keydowns = keydowns        # () -> count of the user's real key-downs so far
        self._secure_input = secure_input
        self._call_after = call_after
        self._note = note                # (title, message) -> None: one user-visible note
        self._log = log
        self._shutdown_requested = shutdown_requested
        self._clock = clock
        self._timer = timer
        self._on_done = on_done          # () -> None: one item left the queue
        self._own_window_front = own_window_front  # () -> bool: Sotto's alert would get the Cmd-V
        self._copy = copy                # (text) -> None: put text on the general clipboard
        self._items: list[tuple] = []
        self._poll_armed = False
        self._waiting = False
        self._blocked_since: float | None = None
        self.last_delivery: dict | None = None  # the insert "scratch that" may undo

    def paste(self, text: str, in_history: bool = True) -> None:
        """in_history is False when History could not keep this dictation
        (F16a/F16b): then no note may send the user to History for it."""
        self._items.append(("paste", text, in_history))
        self._pump()

    def undo(self) -> None:
        self._items.append(("undo",))
        self._pump()

    def _pump(self) -> None:
        while self._items:
            if self._shutdown_requested():  # checked per item: Quit can land mid-pump
                self._discard_all()  # nothing is posted once shutdown is requested
                return
            if self._keys_held():
                self._wait_or_drop()
                return
            if self._own_window_front():
                self._wait_for_own_window()
                return
            self._blocked_since = None
            self._waiting = False
            item = self._items.pop(0)
            try:
                if item[0] == "paste":
                    self._deliver(item[1], item[2])
                else:
                    self._undo()
            except Exception as exc:  # one failed insert must not strand the items behind it
                self._log(f"! {item[0]} failed ({type(exc).__name__}: {str(exc)[:160]})")
                if item[0] == "paste":
                    self._note("Could not paste the dictation",
                               "It is in History. Open History to copy the text." if item[2]
                               else NOT_IN_HISTORY_EITHER)
            finally:
                self._on_done()

    def _discard_all(self) -> None:
        """Drop every waiting item; each one still reports done."""
        items, self._items = self._items, []
        for _item in items:
            self._on_done()

    def _wait_or_drop(self) -> None:
        now = self._clock()
        if self._recording():
            self._blocked_since = None  # the user is dictating: the release will deliver
        elif self._blocked_since is None:
            self._blocked_since = now
        elif now - self._blocked_since > DELIVERY_WAIT_MAX_S:
            dropped = sum(1 for item in self._items if item[0] == "paste")
            unsaved = [item[1] for item in self._items if item[0] == "paste" and not item[2]]
            self._discard_all()
            self._blocked_since = None
            self._waiting = False
            self._log(f"! a key stayed held for {DELIVERY_WAIT_MAX_S:.0f}s — {dropped} dictation(s) "
                      f"not pasted, " + (f"{len(unsaved)} not in History" if unsaved else "kept in History"))
            if unsaved and self._copy_unsaved(unsaved):
                self._note("Dictation copied to the clipboard",
                           "A key was held for too long to paste it, and History could not save it, "
                           "so the text is on the clipboard.")
            elif unsaved:
                self._note("Dictation not pasted",
                           f"A key was held for too long to paste it. {NOT_IN_HISTORY_EITHER}")
            else:
                self._note("Dictation kept in History",
                           "A key was held for too long to paste it. Open History to copy the text.")
            return
        if not self._waiting:
            self._waiting = True
            self._log("  paste deferred — a key is still held")
        self._arm_poll()

    def _copy_unsaved(self, texts: list[str]) -> bool:
        """Put dropped text History could not save on the clipboard: it is the
        only copy (owner decision 2026-10-09). Returns whether it got there.

        Side effects: replaces the general clipboard; logs a failure."""
        try:
            self._copy("\n\n".join(texts))
            return True
        except Exception as exc:
            self._log(f"! could not copy the dropped dictation ({type(exc).__name__})")
            return False

    def _wait_for_own_window(self) -> None:
        """Hold the queue while Sotto's own window is in front (N24); no drop."""
        self._blocked_since = None
        if not self._waiting:
            self._waiting = True
            self._log("  paste deferred — Sotto's own window is in front")
        self._arm_poll()

    def _arm_poll(self) -> None:
        if not self._poll_armed:
            self._poll_armed = True
            timer = self._timer(DELIVERY_POLL_S, self._call_after, (self._resume,))
            timer.daemon = True
            timer.start()

    def _resume(self) -> None:
        self._poll_armed = False
        self._pump()

    def _deliver(self, text: str, in_history: bool = True) -> None:
        pid, keydowns = self._frontmost_pid(), self._keydowns()  # the paste target, before ⌘V
        if self._insert(text):  # a secure-input decline inserted nothing: arm no undo
            self.last_delivery = {"at": self._clock(), "pid": pid, "keydowns": keydowns}
        else:  # N34: every other drop path tells the user, so this one does too
            self._note("Dictation not pasted",
                       "Secure input is on (a password field may have focus), so Sotto did not "
                       "paste. " + ("Open History to copy the text." if in_history else NOT_IN_HISTORY_EITHER))

    def _undo(self) -> None:
        """'scratch that': the app's own ⌘Z, only for Sotto's own recent insert —
        same app still in front, nothing typed by the user since."""
        import voice_commands
        last = self.last_delivery
        if last is None or self._clock() - last["at"] > voice_commands.SCRATCH_WINDOW_S:
            self._log("  scratch that: nothing recent to undo")
            return
        pid = self._frontmost_pid()
        if pid is None or last["pid"] is None:  # no identity to compare: None == None proves nothing
            self._log("  scratch that: the app in front is unknown — nothing undone")
            return
        if pid != last["pid"]:
            self._log("  scratch that: the dictation went to another app — nothing undone")
            return
        if self._keydowns() != last["keydowns"]:
            self._log("  scratch that: you typed since the dictation — nothing undone")
            return
        if self._secure_input():
            return
        self._undo_keys()
        self.last_delivery = None
        self._log("↶ scratch that — undid the last dictation")
