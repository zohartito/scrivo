"""Keeps a plain `python -m unittest discover -s tests` away from real user data (F64).

`scrivo_paths` reads SCRIVO_DATA_DIR / SCRIVO_HF_HOME once, at its first import, and
without them it points at ~/Library/Application Support/sotto or %APPDATA%\\sotto.
Discovery imports test modules in sorted order, so this module loads before any
other test module and points both variables at a throwaway folder first (and
drops their old SOTTO_* names, which scrivo_paths also reads). Every
in-process import of scrivo_paths and every subprocess (they inherit os.environ)
then stays inside that folder. The tests below fail if that ever stops holding.
"""
import atexit
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SCRIVO_PATHS_PRELOADED = "scrivo_paths" in sys.modules
TEST_ROOT = Path(tempfile.mkdtemp(prefix="scrivo-tests-"))
os.environ["SCRIVO_DATA_DIR"] = str(TEST_ROOT / "data")
os.environ["SCRIVO_HF_HOME"] = str(TEST_ROOT / "huggingface")
for _legacy in [name for name in os.environ if name.startswith("SOTTO_")]:
    del os.environ[_legacy]  # Scrivo was called Sotto: scrivo_paths.env falls back to these
atexit.register(shutil.rmtree, TEST_ROOT, ignore_errors=True)


def _under(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def _real_default_folders():
    home = Path.home()
    folders = [home / "Library/Application Support" / name for name in ("sotto", "sotto-alpha", "Sotto")]
    for variable, name in (("APPDATA", "sotto"), ("LOCALAPPDATA", "sotto-alpha")):
        if os.environ.get(variable, "").strip():
            folders.append(Path(os.environ[variable]) / name)
    return folders


class HermeticTestState(unittest.TestCase):
    def test_runs_before_anything_imports_scrivo_paths(self):
        self.assertFalse(SCRIVO_PATHS_PRELOADED, "scrivo_paths was imported before tests/test_00_hermetic.py")

    def test_discovery_imports_this_module_first(self):
        importable = [name for name in sorted(os.listdir(HERE))
                      if (name.startswith("test") and name.endswith(".py")) or (HERE / name / "__init__.py").is_file()]
        self.assertEqual(importable[0], Path(__file__).name)

    def test_data_and_model_cache_are_temporary(self):
        import scrivo_paths
        for path in (scrivo_paths.DATA_DIR, scrivo_paths.MODEL_CACHE_DIR):
            self.assertTrue(_under(path, TEST_ROOT), f"{path} is outside the temporary folder {TEST_ROOT}")
            for real in _real_default_folders():
                self.assertFalse(_under(path, real), f"{path} is inside the real data folder {real}")


if __name__ == "__main__":
    unittest.main()
