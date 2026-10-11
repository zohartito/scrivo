"""The alpha runs without the maintainer lane (restructure step 8, M2).

Each check runs in a fresh interpreter whose import system refuses every
maintainer module: the ``maintainer`` package and the lane's former top-level
names. A passing check proves the alpha path imports none of them. The lane is
reached only through gated edges (``--adaptive``, the lane's learning-*
commands, and History mutations over a data folder the lane has written), and
those must fail closed when the lane's code is missing.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHILD_TIMEOUT_S = 180

# The maintainer package, plus the top-level names the lane's modules had
# before they moved into it (so the blocker means the same on either layout).
MAINTAINER_MODULES = (
    "maintainer", "adaptive_runtime", "adaptive_worker", "calibration_manifest_builder", "calibration_worker",
    "deployment_controller", "inference_scheduler", "launchd_templates", "offline_backup", "sealed_release",
    "silver_experiment", "silver_store", "speech_backends", "teacher_backends", "teacher_consensus",
    "teacher_provision", "teachers",
)

BLOCKER = textwrap.dedent(f"""
    import sys
    sys.path.insert(0, {str(ROOT)!r})
    _LANE = set({MAINTAINER_MODULES!r})

    class _MaintainerBlocker:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in _LANE:
                raise ModuleNotFoundError(f"maintainer module blocked: {{fullname}}", name=fullname)
            return None

    if __BLOCK__:
        sys.meta_path.insert(0, _MaintainerBlocker())
""")

ALPHA_MAC_MODULES = (
    "sotto", "transcription", "app_controller", "pipeline", "delivery", "lifecycle", "gestures", "capture",
    "history", "learning", "adaptive_learning", "ui", "settings_window", "updates", "dictionary", "settings",
    "progress", "voice_commands", "nemotron_backend", "streaming_audio", "vad", "speech_config", "login_item",
    "offline_runtime", "sotto_log", "sotto_paths", "comparator_spool", "evaluation", "audio_codec",
    "storage_lock", "runtime_source_manifest",
)
ALPHA_WINDOWS_MODULES = (
    "sotto_win", "win_asr", "win_capture", "win_hotkey", "win_inject", "win_launch", "win_startup", "win_ui",
    "sotto", "history", "learning", "adaptive_learning", "pipeline", "lifecycle", "gestures", "updates",
    "dictionary", "settings", "progress", "voice_commands", "speech_config", "offline_runtime", "sotto_paths",
    "storage_lock", "audio_codec", "comparator_spool", "evaluation",
)


def run_alpha(code: str, *, block: bool = True, args: tuple = ()) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter (maintainer modules refused unless
    ``block`` is False) with its own temporary data folder, from the repo root."""
    with tempfile.TemporaryDirectory(dir=ROOT) as scratch:
        env = dict(os.environ, SOTTO_DATA_DIR=str(Path(scratch) / "data"), SOTTO_HF_HOME=str(Path(scratch) / "hf"),
                   SOTTO_OFFLINE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", SOTTO_SCRATCH=scratch)
        script = BLOCKER.replace("__BLOCK__", str(block)) + textwrap.dedent(code)
        return subprocess.run([sys.executable, "-c", script, *args],
                              cwd=ROOT, env=env, capture_output=True, text=True, timeout=CHILD_TIMEOUT_S)


def assert_ok(test: unittest.TestCase, done: subprocess.CompletedProcess) -> None:
    test.assertEqual(done.returncode, 0, f"stdout:\n{done.stdout}\nstderr:\n{done.stderr[-4000:]}")


class AlphaImportsTests(unittest.TestCase):
    def modules(self):
        return ALPHA_WINDOWS_MODULES if sys.platform == "win32" else ALPHA_MAC_MODULES

    def test_every_alpha_module_imports_with_the_lane_refused(self):
        assert_ok(self, run_alpha(f"""
            import importlib
            for name in {self.modules()!r}:
                importlib.import_module(name)
        """))

    def test_importing_the_alpha_loads_no_maintainer_module_even_when_present(self):
        done = run_alpha(f"""
            import importlib, json
            for name in {self.modules()!r}:
                importlib.import_module(name)
            print(json.dumps(sorted(name for name in sys.modules if name.split(".")[0] in _LANE)))
        """, block=False)
        assert_ok(self, done)
        self.assertEqual(json.loads(done.stdout.splitlines()[-1]), [])


HISTORY_FLOW = """
    import os
    from pathlib import Path
    import numpy as np
    import dictionary, history, learning, sotto
    from adaptive_learning import AdaptiveLearning

    base = Path(os.environ["SOTTO_SCRATCH"]) / "history"
    store = history.HistoryStore(base)
    coordinator = sotto.history_coordinator(store, learning.LearningStore(base))
    guard = AdaptiveLearning(store.base_dir)
    rows = [store.append(f"hello there {i}", np.zeros(1600, np.float32), .1, "model") for i in range(3)]
    first, second = rows[0]["id"], rows[1]["id"]
    assert sotto.nonadaptive_correct(guard, coordinator, first, "hello their",
                                     expected_revision=rows[0]["revision"]) is not None
    assert store.get(first)["correction"]["text"] == "hello their"
    assert coordinator.enroll(first) is not None
    assert sotto.nonadaptive_revoke_learning(guard, coordinator, first) == 1
    assert sotto.nonadaptive_delete(guard, coordinator, second) is True
    assert store.get(second) is None
    sotto.nonadaptive_clear(guard, coordinator)
    assert store.entries(10) == []
    path = Path(os.environ["SOTTO_SCRATCH"]) / "dictionary.txt"
    assert dictionary.add(dictionary.suggest("I use soto daily", "I use Sotto daily", []), path) == 1
    assert dictionary.apply("ask soto", dictionary.load(path))[0] == "ask Sotto"
    print("history flow ok")
"""


class AlphaHistoryTests(unittest.TestCase):
    def test_correct_learning_delete_clear_and_dictionary_need_no_maintainer_module(self):
        done = run_alpha(HISTORY_FLOW)
        assert_ok(self, done)
        self.assertIn("history flow ok", done.stdout)

    def test_a_folder_the_lane_has_written_refuses_history_changes_without_its_code(self):
        """Fail closed: with a silver ledger present and the lane's code
        missing, Delete and Clear raise before History changes."""
        done = run_alpha("""
            import os
            from pathlib import Path
            import numpy as np
            import history, learning, sotto
            from adaptive_learning import AdaptiveLearning

            base = Path(os.environ["SOTTO_SCRATCH"]) / "history"
            store = history.HistoryStore(base)
            coordinator = sotto.history_coordinator(store, learning.LearningStore(base))
            guard = AdaptiveLearning(store.base_dir)
            rows = [store.append(f"words {i}", np.zeros(1600, np.float32), .1, "model") for i in range(2)]
            (base / "adaptive-learning" / "silver" / "silver.sqlite3").write_bytes(b"")
            assert learning.silver_lane_exists(base)
            for action in (lambda: sotto.nonadaptive_delete(guard, coordinator, rows[0]["id"]),
                           lambda: sotto.nonadaptive_clear(guard, coordinator)):
                try:
                    action()
                except ModuleNotFoundError as exc:
                    assert "maintainer module blocked" in str(exc), exc
                else:
                    raise AssertionError("a History change went through without the lane's revocation")
            assert [row["id"] for row in store.entries(10)] == [row["id"] for row in reversed(rows)]
            print("fail closed ok")
        """)
        assert_ok(self, done)
        self.assertIn("fail closed ok", done.stdout)


@unittest.skipUnless(sys.platform == "darwin", "the Mac menu actions, doctor and run() — macOS-only")
class AlphaMacTests(unittest.TestCase):
    def test_menu_correct_delete_clear_and_check_for_updates_with_real_history(self):
        done = run_alpha("""
            import os
            from pathlib import Path
            from unittest.mock import patch
            import numpy as np
            import history, learning, offline_runtime, sotto, updates
            from adaptive_learning import AdaptiveLearning
            import tests.test_app_controller as fixtures

            base = Path(os.environ["SOTTO_SCRATCH"]) / "history"
            world = fixtures.World()
            world.store = history.HistoryStore(base)
            world.coordinator = sotto.history_coordinator(world.store, learning.LearningStore(base))
            world.dependency_guard = AdaptiveLearning(world.store.base_dir)
            for name in ("nonadaptive_correct", "nonadaptive_delete", "nonadaptive_clear",
                         "nonadaptive_revoke_learning"):
                setattr(world, name, getattr(sotto, name))
            rows = [world.store.append(f"hello there {i}", np.zeros(1600, np.float32), .1, "model")
                    for i in range(2)]
            controller = fixtures.build(world)
            controller.correct(rows[0]["id"], "hello their", rows[0]["revision"])
            assert world.store.get(rows[0]["id"])["correction"]["text"] == "hello their"
            controller.delete(rows[1]["id"])
            assert world.store.get(rows[1]["id"]) is None
            controller.clear()
            assert world.store.entries(10) == []
            result = updates.UpdateCheck("current", "")
            with patch.object(updates, "check", lambda root, *, offline, started_from: result), \\
                 patch.object(offline_runtime, "offline_requested", return_value=False), \\
                 patch.dict(os.environ, {"SOTTO_LAUNCHER": "terminal"}), patch("AppKit.NSWorkspace"):
                controller.check_updates()
            errors = [event for event in world.events if event[:2] == ("ui", "show_error")]
            assert not errors, errors
            assert ("log", "update check: current") in world.events, world.events
            print("menu ok")
        """)
        assert_ok(self, done)
        self.assertIn("menu ok", done.stdout)

    def test_doctor(self):
        done = run_alpha("""
            import subprocess
            from unittest.mock import patch
            import sotto

            probe = subprocess.CompletedProcess([], 0, "48000 Hz", "mic: Fake Mic")
            with patch("ApplicationServices.AXIsProcessTrusted", return_value=True), \\
                 patch.object(sotto, "microphone_permission", return_value=True), \\
                 patch("subprocess.run", return_value=probe):
                sotto.doctor()
            print("doctor ok")
        """)
        assert_ok(self, done)
        self.assertIn("doctor ok", done.stdout)

    def test_one_dictation_through_run_then_quit(self):
        """sotto.run() wired for real (History, learning, the dependency
        guard, AppController, the worker, delivery, teardown) with the OS
        boundaries faked: test_nemotron_pipeline's harness, lane refused."""
        done = run_alpha("""
            import unittest
            import tests.test_nemotron_pipeline as pipeline

            name = "test_streaming_capture_reaches_history_and_delivery_once_then_shuts_down"
            result = unittest.TextTestRunner(verbosity=0).run(unittest.TestSuite([pipeline.PipelineTests(name)]))
            raise SystemExit(0 if result.wasSuccessful() and result.testsRun == 1 else 1)
        """)
        assert_ok(self, done)


CLI = """
    import contextlib, io, os, runpy
    sys.argv = ["sotto.py", *sys.argv[1:]]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            runpy.run_path("sotto.py", run_name="__main__")
    finally:
        print(out.getvalue().replace(os.environ["SOTTO_DATA_DIR"], "<data>"))
"""


@unittest.skipUnless(sys.platform == "darwin", "sotto.py's CLI — macOS-only")
class AlphaCliTests(unittest.TestCase):
    def test_learning_status_list_and_dictionary_read_the_same_without_the_lane(self):
        for command in ("learning-status", "learning-list", "dictionary"):
            with self.subTest(command=command):
                with_lane = run_alpha(CLI, block=False, args=(command,))
                without = run_alpha(CLI, args=(command,))
                assert_ok(self, with_lane)
                assert_ok(self, without)
                self.assertEqual(without.stdout, with_lane.stdout)

    def test_the_lane_commands_and_adaptive_run_refuse_cleanly_without_it(self):
        for command in ("learning-preflight", "learning-review-list", "learning-evaluate", "learning-rollback"):
            with self.subTest(command=command):
                done = run_alpha(CLI, args=(command,))
                self.assertEqual(done.returncode, 2, done.stderr[-2000:])
                self.assertIn("maintainer lane", done.stderr)
                self.assertNotIn("Traceback", done.stderr)
        done = run_alpha("""
            from unittest.mock import patch
            import sotto
            sys.argv = ["sotto.py", "run", "--adaptive"]
            with patch.object(sotto, "run", side_effect=AssertionError("run() reached without the lane")):
                sotto.main()
        """)
        self.assertEqual(done.returncode, 2, done.stderr[-2000:])
        self.assertIn("maintainer lane", done.stderr)

    @unittest.skipUnless((ROOT / "maintainer").is_dir(), "the maintainer lane is not in this copy (alpha archive)")
    def test_with_the_lane_present_adaptive_run_still_reaches_run(self):
        done = run_alpha("""
            from unittest.mock import patch
            import sotto
            assert sotto.maintainer_lane_available()
            sys.argv = ["sotto.py", "run", "--adaptive"]
            with patch.object(sotto, "run") as run:
                sotto.main()
            assert run.call_args.kwargs["adaptive"] is True
            print("adaptive run reached")
        """, block=False)
        assert_ok(self, done)
        self.assertIn("adaptive run reached", done.stdout)



def _local_modules(names: set[str]) -> dict[str, str]:
    """Shipped module name -> archive path, for the archive's .py files."""
    modules = {}
    for name in names:
        if name.endswith(".py"):
            relative = Path(name)
            modules[".".join(relative.with_suffix("").parts)] = name
    return modules


class AlphaArchiveTests(unittest.TestCase):
    def test_the_source_archive_ships_no_maintainer_code_and_imports_none(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            import build_alpha_source
        finally:
            sys.path.remove(str(ROOT / "scripts"))
        with tempfile.TemporaryDirectory(dir=ROOT) as scratch:
            output = Path(scratch) / "alpha.tar.gz"
            first = build_alpha_source.build(output)
            with tarfile.open(output) as archive:
                names = {member.name.removeprefix("sotto-alpha/") for member in archive.getmembers()}
                sources = {name: archive.extractfile("sotto-alpha/" + name).read().decode("utf-8")
                           for name in names if name.endswith(".py")}
            self.assertEqual(build_alpha_source.build(output)["sha256"], first["sha256"])  # deterministic
        self.assertEqual(names, set(getattr(build_alpha_source, "ALPHA_FILES", build_alpha_source.PUBLIC_FILES)))
        shipped_lane = sorted(name for name in names
                              if name.startswith("maintainer/") or Path(name).stem in MAINTAINER_MODULES)
        self.assertEqual(shipped_lane, [])
        modules = _local_modules(names)
        for name, source in sorted(sources.items()):
            tree = ast.parse(source)
            module_level = {id(node) for node in tree.body}
            for node in ast.walk(tree):
                for target in _import_targets(node):
                    head = target.split(".")[0]
                    # Shipped tests never touch the lane. Shared code reaches it
                    # only from inside functions (gated); a module-level import
                    # of a repo module must be shipped too.
                    if name.startswith("tests/"):
                        with self.subTest(file=name, imports=target):
                            self.assertNotIn(head, MAINTAINER_MODULES)
                    if id(node) in module_level and (head in MAINTAINER_MODULES or (ROOT / f"{head}.py").exists()):
                        with self.subTest(file=name, module_level_import=target):
                            self.assertIn(head, modules)


def _import_targets(node) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
        return [node.module]
    return []


if __name__ == "__main__":
    unittest.main()
