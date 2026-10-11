"""The Mac app's menu-bar actions.

These were closures inside ``sotto.run()``; they are methods of one
``AppController`` now, like ``sotto_win.Controller`` on Windows, with every
collaborator passed to ``__init__`` (dependency injection), so tests build
it with fakes. tests/test_app_controller.py pins what each action does.

Imports nothing from sotto: ``run()`` passes in the History helpers that
still live there.
"""
from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path


def start_daemon_thread(target) -> None:
    """Run ``target`` on a new daemon thread (the actions' default ``start_thread``).

    Side effects: starts a thread."""
    threading.Thread(target=target, daemon=True).start()


class AppController:
    """What the menu bar's History items do."""

    def __init__(self, *, shutdown, lifecycle, jobs, store, coordinator, dependency_guard,
                 adaptive_runtime, status_ui, ui_call, deliver_call, pending_deliveries, set_clipboard,
                 current_speech_config, log, history_keep: int, overlay_learning_state,
                 active_learning_history_ids, nonadaptive_correct, nonadaptive_delete, nonadaptive_clear,
                 nonadaptive_revoke_learning, save_audio_copy, start_thread=start_daemon_thread) -> None:
        self.shutdown = shutdown
        self.lifecycle = lifecycle
        self.jobs = jobs
        # History, the learning coordinator and the revocation guard an
        # earlier adaptive session needs; adaptive_runtime is None unless
        # adaptive routing is on.
        self.store = store
        self.coordinator = coordinator
        self.dependency_guard = dependency_guard
        self.adaptive_runtime = adaptive_runtime
        # The menu bar (None under --no-overlay) and the main-thread
        # schedulers: ui_call for the UI, deliver_call for counted deliveries.
        self.status_ui = status_ui
        self.ui_call = ui_call
        self.deliver_call = deliver_call
        self.pending_deliveries = pending_deliveries
        self.set_clipboard = set_clipboard
        self.current_speech_config = current_speech_config
        self.log = log
        # History helpers that live in sotto.py.
        self.history_keep = history_keep
        self.overlay_learning_state = overlay_learning_state
        self.active_learning_history_ids = active_learning_history_ids
        self.nonadaptive_correct = nonadaptive_correct
        self.nonadaptive_delete = nonadaptive_delete
        self.nonadaptive_clear = nonadaptive_clear
        self.nonadaptive_revoke_learning = nonadaptive_revoke_learning
        self.save_audio_copy = save_audio_copy
        # Runs an action's slow part off the main thread; tests run it inline.
        self.start_thread = start_thread

    # -- History ----------------------------------------------------------------

    def refresh_history(self) -> None:
        """Show History's rows, with their learning and review state.

        Side effects: schedules the menu's History list (and, adaptive, its
        stage label) on the main thread."""
        if self.status_ui is not None:
            entries = self.store.entries(limit=self.history_keep)
            overlaid = self.overlay_learning_state(entries, self.active_learning_history_ids(self.coordinator))
            if self.adaptive_runtime is not None:
                states = {row["history_id"]: row["status"] for row in self.adaptive_runtime.review_list()}
                overlaid = [dict(row, adaptive_eligible=self.adaptive_runtime.review_eligible_entry(row),
                                 **({"review_state": states[row["id"]]} if row.get("id") in states else {}))
                            for row in overlaid]
                adaptive_state = self.adaptive_runtime.status()
                active = adaptive_state.get("active_generation")
                if active and active.get("status") in {"pool_closed", "evaluating"}:
                    label = "adaptive: evaluating"
                elif active:
                    label = f"adaptive: pool {active.get('resolved', 0)}/33"
                elif adaptive_state["development"].get("ready"):
                    label = "adaptive: collecting challenger"
                else:
                    label = f"adaptive: dev {adaptive_state['development'].get('verified_nonempty', 0)}/20"
                self.ui_call(self.status_ui.set_adaptive_stage, label)
            self.ui_call(self.status_ui.refresh_history, overlaid)

    def copy_text(self, text: str) -> None:
        """Retry's clipboard copy. The worker schedules it with deliver_call,
        so it finishes its PendingDeliveries entry like a paste (N19).

        Side effects: replaces the clipboard (not after shutdown); finishes
        one pending delivery."""
        try:
            if self.shutdown.requested():
                return
            self.set_clipboard(text)
        finally:
            self.pending_deliveries.finish()

    def copy(self, entry_id: str) -> None:
        """Copy a History transcript.

        Side effects: on a background thread, schedules the counted
        clipboard copy; shows an error if History cannot be read."""
        def work() -> None:
            try:
                entry = self.store.get(entry_id)
                if entry:
                    self.deliver_call(self.copy_text, entry["text"])  # copy_text finishes its count
            except Exception as exc:
                self.ui_call(self.status_ui.show_error, "Could not copy transcript", str(exc)[:160])
        self.start_thread(work)

    def retry(self, entry_id: str) -> None:
        """Refused while Quit, Restart, an update or an engine switch
        drains (N18, the Mac side of F73): new work would keep it waiting.

        Side effects: queues a Retry job, or logs and shows the refusal."""
        refused = self.lifecycle.run_if_open(
            lambda: self.shutdown.enqueue(self.jobs, ("retry", entry_id, time.monotonic(), uuid.uuid4().hex,
                                                      self.current_speech_config())))
        if refused:
            self.log(f"○ Sotto is {refused} — Retry refused")
            self.ui_call(self.status_ui.show_error, "Not retried",
                         f"Sotto is {refused}. Retry once it is running again.")

    def save(self, entry_id: str) -> None:
        """Save a recording's audio to the Desktop.

        Side effects: on a background thread, writes one file to the
        Desktop and logs it; shows an error when that fails."""
        def work() -> None:
            try:
                entry = self.store.get(entry_id)
                if not entry:
                    return
                target = self.save_audio_copy(self.store.audio_path(entry_id), Path.home() / "Desktop",
                                              entry["ts"])
                self.log(f"✓ saved audio to Desktop as {target.name}")
            except FileNotFoundError:
                self.log("! audio missing for that entry")
                self.ui_call(self.status_ui.show_error, "Could not save audio",
                             "The audio file is no longer available.")
            except Exception as exc:
                self.ui_call(self.status_ui.show_error, "Could not save audio", str(exc)[:160])
        self.start_thread(work)

    def delete(self, entry_id: str) -> None:
        """Delete a recording (revoking any learning evidence first).

        Side effects: on a background thread, changes History and the
        learning set, then refreshes the menu; shows an error on failure."""
        def work() -> None:
            if self.shutdown.requested():
                return
            try:
                (self.adaptive_runtime.delete(entry_id) if self.adaptive_runtime is not None else
                 self.nonadaptive_delete(self.dependency_guard, self.coordinator, entry_id))
                self.refresh_history()
            except Exception as exc:
                self.ui_call(self.status_ui.show_error, "Could not delete recording", str(exc)[:160])
        self.start_thread(work)

    def clear(self) -> None:
        """Delete every recording (revoking any learning evidence first).

        Side effects: on a background thread, empties History and the
        learning set, then refreshes the menu; shows an error on failure."""
        def work() -> None:
            if self.shutdown.requested():
                return
            try:
                (self.adaptive_runtime.clear() if self.adaptive_runtime is not None
                 else self.nonadaptive_clear(self.dependency_guard, self.coordinator))
                self.refresh_history()
            except Exception as exc:
                self.ui_call(self.status_ui.show_error, "Could not clear history", str(exc)[:160])
        self.start_thread(work)

    def correct(self, entry_id: str, corrected_text: str, expected_revision: int | None = None) -> None:
        """Save a corrected transcript and offer the dictionary rules it suggests.

        Side effects: on a background thread, changes History (or, for an
        eligible adaptive row, records the review), refreshes the menu and
        shows the result."""
        def work() -> None:
            if self.shutdown.requested():
                return
            try:
                entry = self.store.get(entry_id)
                before = str((entry or {}).get("text") or "")
                eligible = self.adaptive_runtime is not None and self.adaptive_runtime.review_eligible_entry(entry)
                if eligible:
                    self.adaptive_runtime.review(entry_id, "corrected", reference=corrected_text,
                                                 expected_revision=expected_revision)
                elif self.nonadaptive_correct(self.dependency_guard, self.coordinator, entry_id, corrected_text,
                                              expected_revision=expected_revision) is None:
                    raise ValueError("The recording no longer exists")
                self.refresh_history()
                message = ("The corrected transcript and audio are retained locally for private learning."
                           if eligible else
                           "The corrected transcript was saved. Add it to the learning set separately if wanted.")
                import dictionary
                suggestions = dictionary.suggest(before, corrected_text,
                                                 dictionary.load(dictionary.DICTIONARY_PATH))
                if suggestions:
                    self.ui_call(self.status_ui.offer_dictionary_rules,
                                 [(rule.heard, rule.write) for rule in suggestions], message)
                else:
                    self.ui_call(self.status_ui.show_info, "Correction saved", message)
            except Exception as exc:
                self.ui_call(self.status_ui.show_error, "Could not save correction", str(exc)[:160])
        self.start_thread(work)

    def add_rules(self, rules: list) -> None:
        """Confirmed correction suggestions become dictionary rules.

        Side effects: writes the dictionary file, logs, refreshes the menu."""
        try:
            import dictionary
            added = dictionary.add([dictionary.Rule(heard, write) for heard, write in rules],
                                   dictionary.DICTIONARY_PATH)
            self.log(f"dictionary: added {added} rule(s)")
            self.refresh_history()
        except Exception as exc:
            self.ui_call(self.status_ui.show_error, "Could not update the dictionary", str(exc)[:160])

    # -- learning ---------------------------------------------------------------

    def enroll(self, entry_id: str) -> None:
        """Add a corrected recording to the learning set.

        Side effects: on a background thread, changes the learning set,
        refreshes the menu and shows the result."""
        def work() -> None:
            if self.shutdown.requested():
                return
            try:
                snapshot = self.coordinator.enroll(entry_id)
                if snapshot is None:
                    raise ValueError("A current correction is required before enrollment")
                self.refresh_history()
                self.ui_call(self.status_ui.show_info, "Added to learning set",
                             "The corrected audio and transcript are now retained locally.")
            except Exception as exc:
                self.ui_call(self.status_ui.show_error, "Could not enroll sample", str(exc)[:160])
        self.start_thread(work)

    def revoke(self, entry_id: str) -> None:
        """Remove a recording from the learning set.

        Side effects: on a background thread, changes the learning set,
        refreshes the menu and shows how many samples went."""
        def work() -> None:
            if self.shutdown.requested():
                return
            try:
                count = (int(self.adaptive_runtime.revoke(entry_id)) if self.adaptive_runtime is not None else
                         self.nonadaptive_revoke_learning(self.dependency_guard, self.coordinator, entry_id))
                self.refresh_history()
                self.ui_call(self.status_ui.show_info, "Removed from learning set",
                             f"Removed {count} local learning sample(s).")
            except Exception as exc:
                self.ui_call(self.status_ui.show_error, "Could not remove learning sample", str(exc)[:160])
        self.start_thread(work)

    def correct_as_is(self, entry_id: str, expected_revision: int | None = None) -> None:
        """Adaptive review: the transcript is right as it is.

        Side effects: on a background thread, records the review and
        refreshes the menu; shows an error on failure."""
        def work() -> None:
            if self.shutdown.requested():
                return
            try:
                if self.adaptive_runtime is None: raise ValueError("Adaptive English is not enabled")
                self.adaptive_runtime.review(entry_id, "correct_as_is", expected_revision=expected_revision)
                self.refresh_history()
            except Exception as exc:
                self.ui_call(self.status_ui.show_error, "Could not review recording", str(exc)[:160])
        self.start_thread(work)

    def no_speech(self, entry_id: str, expected_revision: int | None = None) -> None:
        """Adaptive review: the recording holds no speech.

        Side effects: on a background thread, records the review and
        refreshes the menu; shows an error on failure."""
        def work() -> None:
            if self.shutdown.requested():
                return
            try:
                if self.adaptive_runtime is None: raise ValueError("Adaptive English is not enabled")
                self.adaptive_runtime.review(entry_id, "no_speech", expected_revision=expected_revision)
                self.refresh_history()
            except Exception as exc:
                self.ui_call(self.status_ui.show_error, "Could not review recording", str(exc)[:160])
        self.start_thread(work)

    def skip_learning_review(self, entry_id: str, reason: str, expected_revision: int | None = None) -> None:
        """Adaptive review: leave this recording out, for ``reason``.

        Side effects: on a background thread, records the review and
        refreshes the menu; shows an error on failure."""
        def work() -> None:
            if self.shutdown.requested():
                return
            try:
                if self.adaptive_runtime is None: raise ValueError("Adaptive English is not enabled")
                self.adaptive_runtime.review(entry_id, "skip", reason=reason, expected_revision=expected_revision)
                self.refresh_history()
            except Exception as exc:
                self.ui_call(self.status_ui.show_error, "Could not review recording", str(exc)[:160])
        self.start_thread(work)
