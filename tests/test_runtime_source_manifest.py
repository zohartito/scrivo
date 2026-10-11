from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import runtime_source_manifest as source


class RuntimeSourceManifestTests(unittest.TestCase):
    def test_runtime_requirements_and_pinned_policy_are_manifest_bound(self):
        self.assertIn("requirements.txt", source.RUNTIME_SOURCE_FILES)
        self.assertIn("runtime_dependency_policy.json", source.RUNTIME_SOURCE_FILES)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "requirements.txt").write_text("root==1\n")
            (root / "runtime_dependency_policy.json").write_text('{"schema":1}\n')
            with patch.object(source, "RUNTIME_SOURCE_FILES", ("requirements.txt", "runtime_dependency_policy.json")):
                first = source.runtime_source_manifest(root)
                (root / "runtime_dependency_policy.json").write_text('{"schema":2}\n')
                self.assertNotEqual(first["digest"], source.runtime_source_manifest(root)["digest"])

    def test_ordered_regular_source_bytes_have_one_unambiguous_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "nested").mkdir()
            (root / "alpha.py").write_bytes(b"alpha")
            (root / "nested" / "beta.py").write_bytes(b"beta")
            with patch.object(source, "RUNTIME_SOURCE_FILES", ("alpha.py", "nested/beta.py")):
                first = source.runtime_source_manifest(root)
                self.assertEqual([item["path"] for item in first["files"]], ["alpha.py", "nested/beta.py"])
                (root / "nested" / "beta.py").write_bytes(b"rotated")
                self.assertNotEqual(first["digest"], source.runtime_source_manifest(root)["digest"])
            (root / "nested" / "beta.py").write_bytes(b"beta")
            with patch.object(source, "RUNTIME_SOURCE_FILES", ("nested/beta.py", "alpha.py")):
                reordered = source.runtime_source_manifest(root)
            self.assertNotEqual(first["digest"], reordered["digest"])

    @unittest.skipUnless((Path(source.__file__).resolve().parent / "maintainer").is_dir(),
                         "the maintainer lane is not in this copy (alpha archive)")
    def test_the_sealed_boundary_covers_the_maintainer_package_and_every_local_import(self):
        """No file the sealed lane can run leaves the boundary: every module in
        maintainer/ is listed, and every repo module a listed file imports (at
        module level or inside a function) is listed too."""
        root = Path(source.__file__).resolve().parent
        listed = set(source.RUNTIME_SOURCE_FILES)
        package = sorted(path.relative_to(root).as_posix() for path in (root / "maintainer").rglob("*.py"))
        self.assertTrue(package)
        self.assertEqual([path for path in package if path not in listed], [])

        def repo_file(name: str) -> str | None:
            parts = name.split(".")
            for end in range(len(parts), 0, -1):
                candidate = Path(*parts[:end])
                for path in (candidate.with_suffix(".py"), candidate / "__init__.py"):
                    if (root / path).is_file():
                        return path.as_posix()
            return None

        missing = set()
        for relative in sorted(name for name in listed if name.endswith(".py")):
            for node in ast.walk(ast.parse((root / relative).read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    targets = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    targets = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
                else:
                    continue
                for target in targets:
                    found = repo_file(target)
                    if found is not None and found not in listed:
                        missing.add(f"{relative}:{node.lineno} imports {found}")
        self.assertEqual(sorted(missing), [])

    def test_missing_unsafe_or_ambiguous_sources_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "ok.py").write_text("ok")
            for files in (("missing.py",), ("../ok.py",), ("ok.py", "ok.py")):
                with self.subTest(files=files), patch.object(source, "RUNTIME_SOURCE_FILES", files):
                    with self.assertRaises(RuntimeError):
                        source.runtime_source_manifest(root)
            outside = root / "outside.py"; outside.write_text("outside")
            (root / "linked.py").symlink_to(outside)
            with patch.object(source, "RUNTIME_SOURCE_FILES", ("linked.py",)):
                with self.assertRaisesRegex(RuntimeError, "unsafe"):
                    source.runtime_source_manifest(root)


if __name__ == "__main__":
    unittest.main()
