import os
from pathlib import Path
import plistlib
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from scripts import bench_common as bench, bench_host


class ModelCacheTests(unittest.TestCase):
    def test_pinned_snapshot_is_local_only_and_accepts_both_mlx_formats(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder) / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").touch()
            fake_hub = SimpleNamespace(snapshot_download=MagicMock(return_value=str(snapshot)))
            for weights in ("weights.safetensors", "weights.npz"):
                with self.subTest(weights=weights), patch.dict("sys.modules", huggingface_hub=fake_hub):
                    (snapshot / weights).touch()
                    with patch.dict(os.environ):
                        self.assertEqual(bench.model_path(Path(folder) / "models"), str(snapshot))
                    self.assertEqual(fake_hub.snapshot_download.call_args.args, (bench.WHISPER_REPO,))
                    self.assertEqual(fake_hub.snapshot_download.call_args.kwargs["revision"], bench.WHISPER_REVISION)
                    self.assertIs(fake_hub.snapshot_download.call_args.kwargs["local_files_only"], True)
                    (snapshot / weights).unlink()

    def test_incomplete_cache_is_refused_without_download_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            fake = SimpleNamespace(snapshot_download=MagicMock(return_value=folder))
            with patch.dict("sys.modules", huggingface_hub=fake), patch.dict(os.environ):
                with self.assertRaisesRegex(bench.BenchError, "not cached"):
                    bench.model_path(folder)
            fake.snapshot_download.assert_called_once()


class BenchHostTests(unittest.TestCase):
    def test_host_is_confined_to_bench_venv(self):
        with tempfile.TemporaryDirectory() as folder:
            venv = Path(folder) / ".venv-bench"
            venv.mkdir()
            (venv / "pyvenv.cfg").touch()
            with patch.object(bench_host.sys, "prefix", str(venv)):
                self.assertEqual(bench_host.host_bundle(), venv / "ScrivoBenchPython.app")
            with patch.object(bench_host.sys, "prefix", folder), self.assertRaises(RuntimeError):
                bench_host.host_bundle()

    def test_setup_copies_interpreter_and_adds_purpose_strings_without_permissions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, destination = root / "interpreter", root / "host.app"
            source.write_bytes(b"stub interpreter")
            foundation = MagicMock()
            foundation.NSBundle.mainBundle.return_value.executablePath.return_value = str(source)
            with (patch.dict("sys.modules", Foundation=foundation),
                  patch.object(bench_host, "host_bundle", return_value=destination),
                  patch.object(bench_host.subprocess, "run") as sign):
                bench_host.setup()
                with self.assertRaises(RuntimeError):
                    bench_host.setup()
            self.assertEqual(source.read_bytes(), b"stub interpreter")
            self.assertEqual((destination / "Contents/MacOS/Python").read_bytes(), b"stub interpreter")
            info = plistlib.loads((destination / "Contents/Info.plist").read_bytes())
            self.assertIn("NSSpeechRecognitionUsageDescription", info)
            self.assertIn("NSMicrophoneUsageDescription", info)
            sign.assert_called_once()
