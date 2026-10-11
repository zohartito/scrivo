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
}
NOT_REEXPORTED = {
    "sotto_log": ("_app_log", "_app_log_lock", "_point_output_at", "_rotate_if_large"),
    "delivery": ("_pending_restore", "_restore_generation", "_secure_input_probe", "_restore_clipboard"),
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

    def test_rebound_module_state_is_not_reexported(self):
        for module_name, names in NOT_REEXPORTED.items():
            for name in names:
                with self.subTest(module=module_name, name=name):
                    self.assertFalse(hasattr(sotto, name))


class NoCycleTests(unittest.TestCase):
    def test_no_split_module_imports_sotto(self):
        for module_name in MOVED:
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
