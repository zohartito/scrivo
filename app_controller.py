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

from lifecycle import app_mode


def start_daemon_thread(target) -> None:
    """Run ``target`` on a new daemon thread (the actions' default ``start_thread``).

    Side effects: starts a thread."""
    threading.Thread(target=target, daemon=True).start()


class AppController:
    """What the menu bar's History and Settings items do."""

    def __init__(self, *, shutdown, lifecycle, jobs, store, coordinator, dependency_guard,
                 adaptive_runtime, status_ui, ui_call, deliver_call, pending_deliveries, set_clipboard,
                 current_speech_config, log, history_keep: int, overlay_learning_state,
                 active_learning_history_ids, nonadaptive_correct, nonadaptive_delete, nonadaptive_clear,
                 nonadaptive_revoke_learning, save_audio_copy, adaptive: bool, use_nemotron: bool,
                 use_parakeet: bool, engine_switching: bool, active_engine: str, settings_locked,
                 binding: dict, rebind, restart, language_lock, language_state: dict, save_language_mode,
                 parakeet_cached, persist_engine_and_restart, run_installer, triggers: dict,
                 trigger_labels: dict, sotto_script: Path, start_thread=start_daemon_thread) -> None:
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
        # Runs an action's slow part off the main thread; tests run it inline.
        self.start_thread = start_thread
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
