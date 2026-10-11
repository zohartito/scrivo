"""The modules split out of sotto.py: sotto still exports every moved public
name as the very same object, and no split module imports sotto (no cycle).

Module state that is rebound at run time (the clipboard restore bookkeeping,
the secure-input probe, the log rotation state) is deliberately not
re-exported: a re-export would be a stale snapshot, so it is read and patched
in its own module.
"""
import ast
import importlib
import unittest
from pathlib import Path

import sotto

ROOT = Path(__file__).resolve().parents[1]

MOVED = {
    "sotto_log": ("keep_app_log_small", "log", "route_app_logs"),
    "delivery": (
        "DELIVERY_POLL_S", "DELIVERY_WAIT_MAX_S", "NOT_IN_HISTORY_EITHER", "RESTORE_DELAY_S",
        "SOTTO_EVENT_TAG", "TRANSIENT_PASTEBOARD_TYPE", "DeliveryQueue", "PendingDeliveries",
        "character_before_caret", "compose_insertion", "flush_clipboard_restore", "frontmost_pid",
        "inject", "is_own_event", "own_window_without_text_field", "post_command_key", "post_own_event",
        "secure_input_active", "set_clipboard_text", "type_text"),
    "lifecycle": (
        "APP_DRAIN_TIMEOUT", "APP_RELAUNCH_SCRIPT", "DRAIN_POLL_S", "LAUNCHD_EXIT_TIMEOUT_S",
        "LAUNCHD_LABEL", "QUIT_DRAIN_S", "RESTART_DRAIN_DEADLINE_S", "RESTART_FAILED_EXIT",
        "RESTART_RELEASE_WAIT_S", "SIGTERM_DRAIN_S", "SIGTERM_KEEP_RESERVE_S", "SIGTERM_RESTORE_WAIT_S",
        "CaptureGate", "Lifecycle", "RestartController", "ShutdownBoundary", "app_mode",
        "commit_if_running", "dictation_in_flight", "end_capture_now", "launchd_owns_this_process",
        "launched_as_sealed_job", "main_thread_dispatch", "relaunch_app_after_exit",
        "shutdown_signal_handlers", "wait_until_idle", "UPDATE_DRAIN_DEADLINE_S"),
    "gestures": ("DOUBLE_TAP_WINDOW_S", "HANDS_FREE_MAX_S", "HOLD_THRESHOLD_S", "GestureEngine"),
    "capture": (
        "HEADPHONE_OUTPUT_TERMINALS", "HEADSET_INPUT_TERMINALS", "HEADSET_NAME_HINTS", "IDLE_STOP_S",
        "PREROLL_S", "RING_S", "SAMPLE_RATE", "SILENCE_RESTART_AFTER_S", "CaptureService",
        "doctor_input_probe", "join_capture_blocks"),
    # Shared with Windows (restructure step 7), which no longer reaches into sotto for it.
    "pipeline": ("finalize_primary_live_delivery",),
    "hotkey": (
        "ALL_MODIFIERS", "KEYCODES", "MODIFIER_FLAGS", "TRIGGER_LABELS", "TRIGGERS", "HotkeyListener",
        "hands_free_hint", "hotkey_release_held_seconds", "parse_hotkey", "trigger_down_in"),
}
NOT_REEXPORTED = {
    "sotto_log": ("_app_log", "_app_log_lock", "_point_output_at", "_rotate_if_large"),
    "delivery": ("_pending_restore", "_restore_generation", "_secure_input_probe", "_restore_clipboard"),
    "lifecycle": (),
    "gestures": (),
    # Private CoreAudio helpers CaptureService reads from capture: without a
    # sotto copy, a patch aimed at sotto fails loudly instead of missing.
    "capture": ("_audio_devices", "_builtin_input_device", "_pin_input_to_builtin", "_select_input_device"),
    "pipeline": (),
    "hotkey": (),
}


class ReexportTests(unittest.TestCase):
    def test_sotto_exports_every_moved_name_as_the_same_object(self):
        for module_name, names in MOVED.items():
            module = importlib.import_module(module_name)
            for name in names:
                with self.subTest(module=module_name, name=name):
                    self.assertIs(getattr(sotto, name), getattr(module, name))

    def test_moved_names_are_defined_in_their_module_not_in_sotto(self):
        sotto_tree = ast.parse((ROOT / "sotto.py").read_text(encoding="utf-8"))
        defined = {node.name for node in sotto_tree.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        defined |= {target.id for node in sotto_tree.body if isinstance(node, ast.Assign)
                    for target in node.targets if isinstance(target, ast.Name)}
        for module_name, names in MOVED.items():
            for name in (*names, *NOT_REEXPORTED[module_name]):
                with self.subTest(module=module_name, name=name):
                    self.assertNotIn(name, defined)
                    self.assertTrue(hasattr(importlib.import_module(module_name), name))

    def test_the_decode_options_live_in_speech_config(self):
        # Shared with Windows (restructure step 7); sotto keeps its old private name.
        import speech_config
        self.assertIs(sotto._transcription_kwargs, speech_config.transcription_kwargs)
        sotto_tree = ast.parse((ROOT / "sotto.py").read_text(encoding="utf-8"))
        self.assertNotIn("_transcription_kwargs", {node.name for node in sotto_tree.body
                                                   if isinstance(node, ast.FunctionDef)})

    def test_rebound_module_state_is_not_reexported(self):
        for module_name, names in NOT_REEXPORTED.items():
            for name in names:
                with self.subTest(module=module_name, name=name):
                    self.assertFalse(hasattr(sotto, name))


class NoCycleTests(unittest.TestCase):
    def test_no_split_module_imports_sotto(self):
        # app_controller.py holds what were run()'s menu closures; run()
        # injects every sotto helper it needs.
        for module_name in (*MOVED, "app_controller"):
            tree = ast.parse((ROOT / f"{module_name}.py").read_text(encoding="utf-8"))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported |= {alias.name.split(".")[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            with self.subTest(module=module_name):
                self.assertNotIn("sotto", imported)


if __name__ == "__main__":
    unittest.main()
