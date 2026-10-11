"""The Mac app's menu-bar actions.

These were closures inside ``sotto.run()``; they are methods of one
``AppController`` now, like ``sotto_win.Controller`` on Windows, with every
collaborator passed to ``__init__`` (dependency injection), so tests build
it with fakes. tests/test_app_controller.py pins what each action does.

Imports nothing from sotto: ``run()`` passes in the History and engine
helpers that still live there.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from pathlib import Path

from lifecycle import UPDATE_DRAIN_DEADLINE_S, app_mode, end_capture_now, wait_until_idle


def start_daemon_thread(target) -> None:
    """Run ``target`` on a new daemon thread (the actions' default ``start_thread``).

    Side effects: starts a thread."""
    threading.Thread(target=target, daemon=True).start()


class AppController:
    """What the menu bar's items do: History, Settings, dictation, restart,
    quit and updates. ``menu_callbacks()`` is what run() hands the menu bar."""

    def __init__(self, *, shutdown, lifecycle, jobs, store, coordinator, dependency_guard,
                 adaptive_runtime, status_ui, ui_call, deliver_call, pending_deliveries, set_clipboard,
                 current_speech_config, log, history_keep: int, overlay_learning_state,
                 active_learning_history_ids, nonadaptive_correct, nonadaptive_delete, nonadaptive_clear,
                 nonadaptive_revoke_learning, save_audio_copy, adaptive: bool, use_nemotron: bool,
                 use_parakeet: bool, engine_switching: bool, active_engine: str, settings_locked,
                 binding: dict, rebind, restart, language_lock, language_state: dict, save_language_mode,
                 parakeet_cached, persist_engine_and_restart, run_installer, triggers: dict,
                 trigger_labels: dict, sotto_script: Path, engine, capture, on_finish,
                 start_thread=start_daemon_thread, wait_until_idle=wait_until_idle) -> None:
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
        # How this run was started: adaptive routing, the engine in use, and
        # whether the engine may be switched; settings_locked names the
        # settings a command-line flag chose for this run.
        self.adaptive = adaptive
        self.use_nemotron = use_nemotron
        self.use_parakeet = use_parakeet
        self.engine_switching = engine_switching
        self.active_engine = active_engine
        self.settings_locked = settings_locked
        # The live trigger/hotkey binding (rebind moves it at the next press)
        # and the restart an engine switch goes through.
        self.binding = binding
        self.rebind = rebind
        self.restart = restart
        # The language choice, shared with run()'s current_speech_config.
        self.language_lock = language_lock
        self.language_state = language_state
        self.save_language_mode = save_language_mode
        # Engine and trigger helpers that live in sotto.py; sotto_script is
        # sotto.py itself (Parakeet's download runs its setup command, and
        # Nemotron's installer sits in its scripts/ folder).
        self.parakeet_cached = parakeet_cached
        self.persist_engine_and_restart = persist_engine_and_restart
        self.run_installer = run_installer
        self.triggers = triggers
        self.trigger_labels = trigger_labels
        self.sotto_script = sotto_script
        # The gesture engine, the microphone capture and the gesture's own
        # finish, for the menu's Start and Finish now.
        self.engine = engine
        self.capture = capture
        self.on_finish = on_finish
        # Runs an action's slow part off the main thread; tests run it inline.
        self.start_thread = start_thread
        # The update restart's drain (lifecycle.wait_until_idle).
        self.wait_until_idle = wait_until_idle
        # Update Now is running (one at a time).
        self.update_running = False
        # The Settings window once opened, and the engine downloads running.
        self.settings_window = None
        self.nemotron_installing = False
        self.parakeet_installing = False

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

    # -- language and engine ------------------------------------------------------

    def set_language(self, mode: str) -> None:
        """Choose the language the next recordings are decoded in.

        Side effects: saves the choice, changes the shared language state,
        updates the menu and logs; shows an error if it cannot be saved."""
        if self.adaptive or self.use_nemotron or self.use_parakeet:
            return
        try:
            selected = self.save_language_mode(mode)
            with self.language_lock:
                self.language_state["mode"] = selected
            self.ui_call(self.status_ui.set_language_mode, selected, True)
            self.log(f"language: {selected} (applies to the next recording)")
        except Exception as exc:
            self.ui_call(self.status_ui.show_error, "Could not change language", str(exc)[:160])

    def set_engine(self, mode: str) -> None:
        """Switch the speech engine: closes the capture gate, saves the
        choice and restarts Sotto; refused while dictation is in flight.

        Side effects: closes the gate (reopened if the switch fails),
        persists the engine and requests the restart; shows any refusal."""
        if self.adaptive or not self.engine_switching or mode == self.active_engine:
            return
        if self.lifecycle.busy():
            self.ui_call(self.status_ui.show_error, "Finish dictation first",
                         "Switch engines after the current recording and transcription finish.")
            return
        # The restart discards a capture in progress: refuse new ones
        # first, then check again, as the update restart does (N8).
        if not self.lifecycle.close("switching engines"):
            self.ui_call(self.status_ui.show_error, "Could not change engine",
                         f"Sotto is {self.lifecycle.closed or 'stopping'}.")
            return

        def failed(message: str) -> None:
            self.lifecycle.reopen("switching engines")
            self.ui_call(self.status_ui.show_error, "Could not change engine", message[:160])
        try:
            if mode == "nemotron":
                from nemotron_backend import installation
                installation()  # verify before persisting a restart choice
            if mode == "parakeet" and not self.parakeet_cached():
                raise RuntimeError("Download Parakeet first: Settings → Download Parakeet.")
            if self.lifecycle.busy():
                raise RuntimeError("Finish the current dictation before switching engines.")
            self.persist_engine_and_restart(mode, self.restart.request, on_failure=failed)
        except Exception as exc:
            failed(str(exc))

    # -- Settings window ----------------------------------------------------------

    def open_dictionary(self) -> None:
        """Open the personal dictionary in the text editor.

        Side effects: creates the dictionary file if missing; starts open(1)."""
        try:
            import dictionary
            import subprocess
            path = dictionary.ensure_file(dictionary.DICTIONARY_PATH)
            subprocess.Popen(["/usr/bin/open", "-t", str(path)])
        except Exception as exc:
            self.ui_call(self.status_ui.show_error, "Could not open the dictionary", str(exc)[:160])

    def open_settings(self) -> None:
        """Show the Settings window, made on first use.

        Side effects: creates and shows an AppKit window."""
        if self.settings_window is None:
            from settings_window import SettingsController
            self.settings_window = SettingsController.alloc().initWithModel_(SettingsModel(self))
        self.settings_window.show()

    def settings_state(self) -> dict:
        """What the Settings window shows."""
        import dictionary
        import login_item
        import settings as store
        from nemotron_backend import is_installed
        from speech_config import ENGINE_CHOICES, WHISPER_LANGUAGES, automatic_languages
        preferences = store.load(store.SETTINGS_PATH)
        return {
            "settings": preferences, "trigger": self.binding["trigger"],
            "hotkey": self.binding["hotkey_spec"], "locked": set(self.settings_locked),
            "trigger_labels": {name: self.trigger_labels[name] for name in store.TRIGGER_KEYS
                               if name in self.triggers},
            "engine_labels": dict(ENGINE_CHOICES),
            "engine": self.active_engine,
            "engine_switching": not self.adaptive and self.engine_switching,
            "nemotron_installed": is_installed(),
            "nemotron_installing": self.nemotron_installing,
            "parakeet_installed": self.parakeet_cached(),
            "parakeet_installing": self.parakeet_installing,
            "fast_available": not self.use_nemotron and not self.adaptive,
            "language_names": WHISPER_LANGUAGES,
            "automatic_languages": list(automatic_languages(preferences["languages"])),
            "login_enabled": login_item.enabled(),
            "login_available": app_mode() and bool(os.environ.get("SOTTO_APP_EXECUTABLE")),
            "dictionary_rules": len(dictionary.load(dictionary.DICTIONARY_PATH)),
            "data_dir": str(store.DATA_DIR),
        }

    def change_setting(self, key: str, value) -> str | None:
        """Apply one change from the Settings window; returns why it was
        refused, or None.

        Side effects: saves the setting; may switch the engine, change the
        login item, move the trigger or hotkey, or refresh the menu; logs."""
        import settings as store
        if key in self.settings_locked:
            return "That setting was chosen with a command-line flag for this run."
        try:
            if key == "engine":
                self.set_engine(value)
                return None
            if key == "launch_at_login":
                import login_item
                executable = os.environ.get("SOTTO_APP_EXECUTABLE", "")
                if value and not (app_mode() and executable):
                    return "Install Sotto as an app first (scripts/install-mac.sh)."
                login_item.enable(executable) if value else login_item.disable()
            store.save({key: value}, store.SETTINGS_PATH)
            if key == "trigger":
                self.rebind(new_trigger=value)
            elif key == "hotkey":
                self.rebind(new_hotkey_spec=value)
            elif key == "languages":
                self.status_ui.automatic_languages = list(value)
                self.refresh_history()
            self.log(f"settings: {key} changed")
            return None
        except Exception as exc:
            return str(exc)

    def settings_action(self, name: str) -> None:
        """A Settings window button.

        Side effects: opens the dictionary or the data folder, or starts an
        engine download."""
        import subprocess
        if name == "open_dictionary":
            self.open_dictionary()
        elif name == "show_data_folder":
            from sotto_paths import DATA_DIR
            subprocess.Popen(["/usr/bin/open", str(DATA_DIR)])
        elif name == "install_nemotron" and not self.nemotron_installing:
            self.install_nemotron()
        elif name == "install_parakeet" and not self.parakeet_installing:
            self.install_parakeet()

    def install_nemotron(self) -> None:
        """Install Nemotron in the background; refused in offline mode.

        Side effects: on a background thread, runs the installer (about
        700 MB), refreshes the menu and an open Settings window, logs and
        shows the result."""
        from offline_runtime import offline_requested
        if offline_requested():
            self.ui_call(self.status_ui.show_error, "Offline mode is on",
                         "Installing Nemotron downloads about 700 MB; start Sotto without the offline flags.")
            return
        self.nemotron_installing = True

        def work() -> None:
            script = self.sotto_script.parent / "scripts" / "setup_nemotron.py"
            self.log("● installing Nemotron (about 700 MB) ...")
            try:
                ok, detail = self.run_installer([sys.executable, str(script)])
            finally:
                self.nemotron_installing = False
            self.refresh_history()
            if self.settings_window is not None:
                self.ui_call(self.settings_window.refresh)
            if ok:
                self.log("✓ Nemotron installed")
                self.ui_call(self.status_ui.show_info, "Nemotron installed",
                             "Choose Speech engine → Nemotron (English streaming) to use it.")
            else:
                self.log(f"! Nemotron install failed ({detail[:80]})")
                self.ui_call(self.status_ui.show_error, "Could not install Nemotron", detail)
        self.start_thread(work)

    def install_parakeet(self) -> None:
        """Download Parakeet in the background; refused in offline mode.

        Side effects: on a background thread, runs sotto.py setup (about
        2.5 GB), refreshes the menu and an open Settings window, logs and
        shows the result."""
        from offline_runtime import offline_requested
        if offline_requested():
            self.ui_call(self.status_ui.show_error, "Offline mode is on",
                         "Downloading Parakeet (2.5 GB) needs the network; start Sotto without the offline flags.")
            return
        self.parakeet_installing = True

        def work() -> None:
            self.log("● downloading Parakeet (about 2.5 GB) ...")
            try:
                ok, detail = self.run_installer([sys.executable, str(self.sotto_script),
                                                 "setup", "--profile", "parakeet"])
            finally:
                self.parakeet_installing = False
            self.refresh_history()
            if self.settings_window is not None:
                self.ui_call(self.settings_window.refresh)
            if ok:
                self.log("✓ Parakeet downloaded")
                self.ui_call(self.status_ui.show_info, "Parakeet downloaded",
                             "Choose Speech engine → Parakeet to use it.")
            else:
                self.log(f"! Parakeet download failed ({detail[:80]})")
                self.ui_call(self.status_ui.show_error, "Could not download Parakeet", detail)
        self.start_thread(work)

    # -- dictation, restart and quit ---------------------------------------------------

    def finish_now(self) -> None:
        """Menu escape hatch — always works, no gesture required.

        Side effects: ends the recording in progress (even one the gesture
        engine lost) so it is transcribed; logs."""
        if self.shutdown.requested():
            return
        ended = end_capture_now(self.engine, self.capture, self.on_finish)
        if ended == "gesture":
            self.log("● finished via menu")
        elif ended == "orphan":
            self.log("● orphan capture finished via menu")

    def start_now(self) -> None:
        """Hands-free from the menu; refused while the gate is closed (N3).
        A start that races the gate closing is refused by on_start.

        Side effects: starts a hands-free recording; logs."""
        if self.shutdown.requested():
            return
        if self.lifecycle.closed:
            self.log(f"○ Sotto is {self.lifecycle.closed} — dictation not started")
            return
        if self.engine.force_start():
            self.log("● dictation started via menu (hands-free)")

    def restart_when_idle(self) -> None:
        """The menu's Restart: finish the dictation in flight like Quit,
        then restart (N12).

        Side effects: closes the gate and drains on a thread, then restarts;
        logs."""
        if self.lifecycle.restart(self.restart_now):
            self.log("● restart requested from the menu — after the current dictation")
        else:
            self.log(f"○ Sotto is {self.lifecycle.closed or 'stopping'} — restart not requested")

    def restart_now(self, after_failure=None) -> None:
        """Restart this foreground process or its own sealed supervisor.
        after_failure runs if the restart does not happen.

        Side effects: requests the restart (shutdown follows); logs, or
        shows why it did not happen."""
        def failed(message: str) -> None:
            if after_failure is not None:
                after_failure()
            self.ui_call(self.status_ui.show_error, "Could not restart Sotto", message[:160])
        try:
            if not self.restart.request(on_failure=failed):
                raise RuntimeError("Sotto is already shutting down")
            self.log("● restarting Sotto")
        except Exception as exc:
            failed(str(exc))

    def quit(self) -> None:
        """Quit once the dictation in progress is done, waiting at most
        QUIT_DRAIN_S (Lifecycle.quit; Ctrl-C and SIGTERM stop the same
        way). Side effects: refuses new recordings, ends the recording in
        progress (even one the gesture engine lost), waits until it is
        pasted, then requests shutdown."""
        self.lifecycle.quit()

    # -- updates ---------------------------------------------------------------------------

    def apply_update(self) -> None:
        """Update this checkout with the installer, then restart into it.

        Side effects: on a background thread, runs the installer's update;
        then closes the gate, drains (at most UPDATE_DRAIN_DEADLINE_S) and
        restarts, or shows why the update did not finish; logs."""
        if not app_mode():
            return  # [N28] its installer rebuilds Sotto.app; a terminal run is given the command
        if self.update_running:
            return
        if self.lifecycle.busy():
            self.ui_call(self.status_ui.show_error, "Finish dictation first",
                         "Update after the current recording and transcription finish.")
            return
        self.update_running = True
        self.log("● updating Sotto")

        def work() -> None:
            import updates
            from sotto_paths import DATA_DIR
            finished, tail, source_changed = updates.apply_mac(
                self.sotto_script.parent, sys.executable, DATA_DIR / "logs" / "update.log")
            self.update_running = False
            if finished:
                self.log("✓ update installed — restarting once dictation is done")
                # A restart discards any recording or transcription in progress,
                # and the user may have dictated while the update ran. Refuse new
                # recordings first, then drain, so none can start in between.
                if not self.lifecycle.close("updating"):
                    self.log(f"○ Sotto is {self.lifecycle.closed or 'stopping'}; the update applies at the next start")
                    return
                if not self.wait_until_idle(lambda: self.lifecycle.busy() and self.lifecycle.closed == "updating",
                                            deadline_s=UPDATE_DRAIN_DEADLINE_S):
                    self.log(f"! dictation still running after {UPDATE_DRAIN_DEADLINE_S:.0f}s — "
                             "restarting into the update anyway")
                if self.lifecycle.closed == "updating":  # else a Quit took over
                    self.restart_now(after_failure=lambda: self.lifecycle.reopen("updating"))
            elif source_changed:
                self.log("! update installed but its setup did not finish (see logs/update.log)")
                self.ui_call(self.status_ui.show_error, "Update not finished",
                             f"{tail}\n\nThe new version is in place but its setup did not finish. "
                             "Run scripts/install-mac.sh --update in Terminal to finish it, then "
                             "choose Restart Sotto. Details are in logs/update.log in the data folder.")
            else:
                self.log("! update failed (see logs/update.log)")
                self.ui_call(self.status_ui.show_error, "Update failed",
                             f"{tail}\n\nThe source was not switched, so Sotto keeps running the "
                             "current version. Details are in logs/update.log in the data folder.")
        self.start_thread(work)

    def check_updates(self) -> None:
        """Contact GitHub only now, because the user asked.

        Side effects: on a background thread, runs git against GitHub and
        logs the result; on the main thread, shows it (offering Update Now
        to Sotto.app, or the Releases page to a copy without git)."""
        def work() -> None:
            import updates
            from offline_runtime import offline_requested
            result = updates.check(self.sotto_script.parent, offline=offline_requested())
            self.log(f"update check: {result.state}" + (f", {result.behind} behind" if result.behind else ""))

            def present() -> None:
                # [N28] Only Sotto.app updates itself; a terminal run is given the command.
                manual = ("run scripts/install-mac.sh --update in Terminal from the Sotto folder."
                          if app_mode() else updates.mac_manual_update(self.sotto_script.parent))
                title, text = updates.describe(result, manual, self_update=app_mode())
                if result.state == "available" and app_mode():
                    if self.status_ui.ask(title, text, "Update Now", "Later"):
                        self.apply_update()
                elif result.state == "not-git":
                    if self.status_ui.ask("Download the latest release", result.detail,
                                          "Open Releases Page", "Close"):
                        import AppKit
                        AppKit.NSWorkspace.sharedWorkspace().openURL_(
                            AppKit.NSURL.URLWithString_(updates.RELEASES_URL))
                elif result.state == "failed":
                    self.status_ui.show_error(title, text)
                else:
                    self.status_ui.show_info(title, text)
            self.ui_call(present)
        self.start_thread(work)

    # -- the menu bar -------------------------------------------------------------------------

    def only_while_running(self, callback):
        """``callback`` as a menu item: once shutdown is requested it
        returns None without running."""
        def wrapped(*args, **kwargs):
            if self.shutdown.requested():
                return None
            return callback(*args, **kwargs)
        return wrapped

    def menu_callbacks(self) -> dict:
        """The menu bar's items (ui.StatusUI.set_history_callbacks). Every
        item but Quit is inert once shutdown is requested."""
        return {
            "quit": self.quit,
            "restart": self.only_while_running(self.restart_when_idle),
            "copy": self.only_while_running(self.copy), "retry": self.only_while_running(self.retry),
            "save": self.only_while_running(self.save), "delete": self.only_while_running(self.delete),
            "clear": self.only_while_running(self.clear), "correct": self.only_while_running(self.correct),
            "enroll": self.only_while_running(self.enroll), "revoke": self.only_while_running(self.revoke),
            "correct_as_is": self.only_while_running(self.correct_as_is),
            "no_speech": self.only_while_running(self.no_speech),
            "skip_learning_review": self.only_while_running(self.skip_learning_review),
            "finish_now": self.only_while_running(self.finish_now),
            "start_now": self.only_while_running(self.start_now),
            "set_language": self.only_while_running(self.set_language),
            "set_engine": self.only_while_running(self.set_engine),
            "add_rules": self.only_while_running(self.add_rules),
            "open_dictionary": self.only_while_running(self.open_dictionary),
            "open_settings": self.only_while_running(self.open_settings),
            "check_updates": self.only_while_running(self.check_updates),
        }


class SettingsModel:
    """What the Settings window reads and changes (main thread only):
    settings_window.SettingsController's model, backed by the controller."""

    def __init__(self, controller: AppController) -> None:
        self.controller = controller

    def state(self) -> dict:
        return self.controller.settings_state()

    def change(self, key: str, value) -> str | None:
        return self.controller.change_setting(key, value)

    def action(self, name: str) -> None:
        self.controller.settings_action(name)
