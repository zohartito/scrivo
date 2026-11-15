import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts import bench_common as bench, record_bench_clip as recorder


class ManifestAppendTests(unittest.TestCase):
    def test_append_preserves_human_reference_and_only_adds_empty_slot(self):
        existing = {"id": "c01", "audio": "audio/c01.wav", "reference": "PRIVATE-MARKER", "required_terms": []}
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as handle:
            original = json.dumps(existing)
            handle.write(original)  # Also cover a manifest without its last newline.
            recorder.append_entry(handle, "c02", "en")
            handle.seek(0)
            text = handle.read()
        self.assertTrue(text.startswith(original + "\n"))
        second = json.loads(text.splitlines()[1])
        self.assertEqual(second, {"id": "c02", "audio": "audio/c02.wav", "reference": "", "language": "en",
                                  "tags": [], "required_terms": []})

    def test_duplicate_corrupt_manifest_and_invalid_id_are_not_rewritten(self):
        for content, clip_id in (("PRIVATE-MARKER", "c01"), ('{"id":"c01"}', "c01"), ("", "../c01")):
            with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as handle:
                handle.write(content)
                with self.assertRaises(bench.BenchError):
                    recorder.append_entry(handle, clip_id, "en")
                handle.seek(0)
                self.assertEqual(handle.read(), content)

    def test_repo_destinations_are_refused(self):
        with self.assertRaises(bench.BenchError):
            bench.outside_repo(bench.ROOT / "private-recording")

    @unittest.skipUnless(os.name == "posix", "Recorder uses macOS file locking")
    def test_symlinked_manifest_is_not_followed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / "original"
            original.write_text("PRIVATE-MARKER")
            (root / "manifest.jsonl").symlink_to(original)
            with self.assertRaises(OSError), recorder.locked_manifest(root):
                self.fail("must not enter")
            self.assertEqual(original.read_text(), "PRIVATE-MARKER")

    def test_capture_cleanup_on_cancel_uses_fakes_only(self):
        fake_capture = MagicMock()
        capture = fake_capture.CaptureService.return_value
        capture.is_waking.return_value = False
        capture.is_live.return_value = True
        fake_codec = MagicMock()
        with patch.dict("sys.modules", {"capture": fake_capture, "audio_codec": fake_codec}):
            with (patch.object(recorder.time, "sleep", side_effect=KeyboardInterrupt),
                  self.assertRaises(KeyboardInterrupt)):
                recorder.record_audio(1, lambda: None)
        capture.shutdown.assert_called_once()
        capture.wait_released.assert_called_once_with(5)
        fake_codec.encode_canonical.assert_not_called()

    def test_invalid_cli_never_opens_microphone(self):
        with patch.object(recorder, "record_audio") as record, patch("sys.stderr", io.StringIO()):
            self.assertEqual(recorder.main(["--id", "../PRIVATE-MARKER", "--dir", "PRIVATE-MARKER"]), 1)
        record.assert_not_called()
