from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from evaluation import aggregate_results, evaluate_sample
from scripts import bench_common as bench, own_voice_bench
from tests.test_platform_baseline import BenchFixture, MARKER


class OwnVoiceTests(BenchFixture):
    def setUp(self):
        super().setUp()
        self.stack.enter_context(patch.object(own_voice_bench, "ensure_host"))
        self.platform = self.folder / "companion.md"

    def run_own(self, extra=()):
        return self.invoke(["--platform-out", str(self.platform), *extra], main=own_voice_bench.main)

    def test_both_reports_share_one_pass_and_contain_aggregates_only(self):
        self.assertEqual(self.run_own()[0], 0)
        own, platform = self.output.read_text(), self.platform.read_text()
        self.assertNotIn(MARKER, own + platform)
        self.assertIn("wer [95% CI]", own)
        self.assertIn("Paired WER difference", own)
        self.assertIn("not a macOS Dictation product test", own)
        self.assertEqual(self.whisper.call_count, 10)
        self.assertEqual(self.apple.transcribe.call_count, 10)
        self.assertEqual(len(platform.splitlines()), 4)

    def test_dry_run_missing_reference_and_backend_failure_write_neither_report(self):
        self.assertEqual(self.run_own(["--dry-run"])[0], 0)
        self.apple_class.assert_not_called()
        self.entries[0]["reference"] = ""
        self.save_manifest()
        self.assertEqual(self.run_own()[0], 1)
        self.entries[0]["reference"] = f"{MARKER} term"
        self.save_manifest()
        self.apple.transcribe.side_effect = RuntimeError(MARKER)
        self.assertEqual(self.run_own()[0], 1)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.platform.exists())

    def test_output_alias_and_existing_companion_stop_before_recognition(self):
        self.platform = self.output
        self.assertEqual(self.run_own()[0], 1)
        self.platform = self.folder / "companion.md"
        self.platform.write_text("keep")
        self.assertEqual(self.run_own()[0], 1)
        self.assertEqual(self.platform.read_text(), "keep")
        self.apple_class.assert_not_called()

    def test_second_write_failure_removes_only_new_reports(self):
        original_open = Path.open

        def fail_companion(path, *args, **kwargs):
            if path == self.platform:
                raise OSError(MARKER)
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", fail_companion):
            self.assertEqual(self.run_own()[0], 1)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.platform.exists())


class BootstrapTests(unittest.TestCase):
    @staticmethod
    def measurement(scores):
        return bench.Measurement("stub", aggregate_results(scores)["overall"], scores, [1., 3.])

    def test_micro_weighting_median_pairing_and_reproducibility(self):
        # Deliberately uneven reference lengths distinguish micro from macro WER.
        scores = [evaluate_sample("a", "b"), evaluate_sample("a b c", "a b c")]
        first = self.measurement(scores)
        self.assertEqual(own_voice_bench.statistics_for(first, [0, 1]), (0.25, 0.25, None, 2.))
        cis, difference = own_voice_bench.intervals([first, first], replicates=100)
        self.assertIsNone(cis[0][2])
        self.assertEqual(difference, (0, 0))
        self.assertEqual((cis, difference), own_voice_bench.intervals([first, first], replicates=100))

    def test_unpaired_or_empty_bootstrap_is_refused(self):
        first = self.measurement([evaluate_sample("a", "a"), evaluate_sample("b", "b")])
        second = self.measurement([])
        with self.assertRaises(ValueError):
            own_voice_bench.intervals([first, second])


class ConsolePrivacyTests(unittest.TestCase):
    def test_python_and_native_backend_output_are_discarded_in_subprocess(self):
        done = subprocess.run(
            [sys.executable, "-m", "unittest",
             "tests.test_platform_baseline.PlatformBaselineTests.test_exact_schema_same_clips_and_no_private_output"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn(MARKER, done.stdout + done.stderr)

    def test_argument_errors_do_not_echo_private_values(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), self.assertRaises(SystemExit):
            own_voice_bench.main([MARKER, "--out", MARKER, "--platform-out", MARKER, "--unknown", MARKER])
        self.assertNotIn(MARKER, out.getvalue() + err.getvalue())
