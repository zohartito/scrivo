from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from evaluation import aggregate_results, evaluate_sample
from scripts import bench_common as bench, own_voice_bench
if __package__:
    from .test_platform_baseline import BenchFixture, MARKER
else:  # unittest discovery imports modules directly (also on Windows).
    from test_platform_baseline import BenchFixture, MARKER


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

    def test_three_backends_score_same_clips_with_intervals_and_paired_difference(self):
        # Nonidentical hypotheses exercise the real scoring and bootstrap path.
        for row in self.entries:
            row["reference"] = "a b c d"
            row["required_terms"] = ["d"]
        self.save_manifest()
        self.apple.transcribe.side_effect = lambda _: "a b x y"
        self.whisper.side_effect = lambda *_: "a b c x"
        with patch.object(bench, "parakeet_model_path", return_value="/private-parakeet"), \
             patch.object(bench, "Parakeet") as backend, \
             patch.object(bench.time, "perf_counter", side_effect=range(60)):
            backend.return_value.transcribe.side_effect = lambda _: "a b c d"
            self.assertEqual(self.run_own(["--with-parakeet"])[0], 0)
        backend.assert_called_once_with("/private-parakeet")
        self.assertEqual([call.args[0] for call in backend.return_value.transcribe.call_args_list],
                         [call.args[0] for call in self.apple.transcribe.call_args_list])
        self.assertEqual(self.whisper.call_count, 10)
        own, platform = self.output.read_text(), self.platform.read_text()
        self.assertEqual(len(platform.splitlines()), 5)
        self.assertIn("| 0.500000 [0.500000, 0.500000]", own)
        self.assertIn("| 0.250000 [0.250000, 0.250000]", own)
        self.assertIn(f"| {bench.PARAKEET_BACKEND} | 10 | 0.000000 [0.000000, 0.000000] | "
                      "0.000000 [0.000000, 0.000000] | 1.000000 [1.000000, 1.000000] | "
                      "1.000000 [1.000000, 1.000000] |", own)
        self.assertIn("Paired WER difference (Parakeet minus Whisper): "
                      "-0.250000 [-0.250000, -0.250000]", own)
        self.assertIn("Negative favors Parakeet; positive favors Whisper.", own)
        self.assertIn("Qwen3-ASR and Nemotron were not benchmarked.", own)
        self.assertNotIn("Nemotron and Parakeet were not benchmarked", own)
        self.assertIn(bench.PARAKEET_REVISION, own)
        self.assertIn("Apple first, then Whisper, then Parakeet", own)
        self.assertIn("Parakeet detects language", own)

    def test_numeric_form_audit_uses_real_scoring_path_and_only_retains_counts(self):
        # Fixed transcript mutations, not recognition measurements or self-equality.
        cases = (("1/8", "1.8"), ("1:50", "1.50"), ("1 1/8", "1.1/8"),
                 ("1/8 then 1:50", "1:50 then 1/8"), ("1/8", "1/8 and 2"),
                 ("1/8", "1 1/8"), ("1.25", "(1.25)!"), ("12:30:45", "(12:30:45)!"),
                 ("ordinary prose", "different prose"), ("one eighth", "1/8"))
        for row, (reference, _) in zip(self.entries, cases, strict=True):
            row.update(reference=f"{MARKER} {reference}", required_terms=[])
        self.save_manifest()
        measured_runs = []
        original_measure = bench.measure

        def collect(*args, **kwargs):
            measured = original_measure(*args, **kwargs)
            measured_runs.append(measured)
            return measured

        with patch.object(bench, "measure", side_effect=collect):
            for enabled in (False, True):
                self.apple.transcribe.side_effect = [f"{MARKER} {hyp}" for _, hyp in cases]
                self.whisper.side_effect = [f"{MARKER} ({ref})!" for ref, _ in cases]
                self.output = self.folder / f"own-{enabled}.md"
                self.platform = self.folder / f"platform-{enabled}.md"
                with patch.object(bench.time, "perf_counter", side_effect=range(40)):
                    self.assertEqual(self.run_own(["--numeric-form-audit"] if enabled else [])[0], 0)
        baseline, audited = measured_runs
        self.assertEqual([result.numeric_form_audit for result in baseline], [None, None])
        self.assertEqual([result.numeric_form_audit for result in audited], [(2, 8), (8, 8)])
        for before, after in zip(baseline, audited, strict=True):
            self.assertEqual((before.overall, before.samples, before.latencies),
                             (after.overall, after.samples, after.latencies))
        # All three punctuation corruptions are invisible to baseline WER/CER.
        self.assertEqual([(row.wer, row.cer) for row in audited[0].samples[:3]], [(0, 0)] * 3)
        own = self.output.read_text()
        self.assertIn("Numeric-form audit", own)
        self.assertIn(f"| {bench.BACKENDS[0]} | 2 | 8 |", own)
        self.assertIn(f"| {bench.BACKENDS[1]} | 8 | 8 |", own)
        self.assertNotIn("Numeric-form audit", (self.folder / "own-False.md").read_text())
        self.assertEqual((self.folder / "platform-False.md").read_bytes(), self.platform.read_bytes())
        for private in (MARKER, "1/8", "1:50", "1.25", str(self.folder)):
            self.assertNotIn(private, own + self.platform.read_text() + repr(audited))

    def test_numeric_form_audit_no_eligible_clips_is_na_and_dry_run_does_not_score(self):
        with patch.object(bench, "measure", wraps=bench.measure) as measure:
            self.assertEqual(self.run_own(["--numeric-form-audit", "--dry-run"])[0], 0)
            measure.assert_not_called()
        self.assertEqual(self.run_own(["--numeric-form-audit"])[0], 0)
        self.assertIn(f"| {bench.BACKENDS[0]} | n/a | 0 |", self.output.read_text())

    def test_numeric_form_audit_is_not_a_platform_option(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            bench.parser("platform").parse_args(["manifest", "--numeric-form-audit"])
        self.assertEqual(raised.exception.code, 2)

    def test_numeric_form_audit_includes_optional_backend(self):
        for row in self.entries:
            row.update(reference=f"{MARKER} 1/8", required_terms=[])
        self.save_manifest()
        self.apple.transcribe.side_effect = lambda _: f"{MARKER} 1.8"
        self.whisper.side_effect = lambda *_: f"{MARKER} (1/8)!"
        with patch.object(bench, "parakeet_model_path", return_value="/private-parakeet"), \
             patch.object(bench, "Parakeet") as backend:
            backend.return_value.transcribe.return_value = f"{MARKER} 1 1/8"
            self.assertEqual(self.run_own(["--with-parakeet", "--numeric-form-audit"])[0], 0)
        own = self.output.read_text()
        for label, matched in zip((*bench.BACKENDS, bench.PARAKEET_BACKEND), (0, 10, 0), strict=True):
            self.assertIn(f"| {label} | {matched} | 10 |", own)
        self.assertNotIn(MARKER, own)

    def test_optional_cache_failure_is_redacted_and_dry_run_never_resolves_it(self):
        with patch.object(bench, "parakeet_model_path",
                          side_effect=bench.BenchError("Parakeet is not cached")) as cache:
            self.assertEqual(self.run_own(["--with-parakeet", "--dry-run"])[0], 0)
            cache.assert_not_called()
            self.assertEqual(self.run_own(["--with-parakeet"])[0], 1)
        self.apple_class.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.platform.exists())

    def test_parakeet_decode_failure_writes_neither_report(self):
        with patch.object(bench, "parakeet_model_path", return_value="/private-parakeet"), \
             patch.object(bench, "Parakeet") as backend:
            backend.return_value.transcribe.side_effect = RuntimeError(MARKER)
            self.assertEqual(self.run_own(["--with-parakeet"])[0], 1)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.platform.exists())

    def test_default_never_resolves_or_constructs_parakeet(self):
        with patch.object(bench, "parakeet_model_path") as cache, patch.object(bench, "Parakeet") as backend, \
             patch.object(bench, "numeric_expression_exact_match") as audit:
            self.assertEqual(self.run_own()[0], 0)
        cache.assert_not_called()
        backend.assert_not_called()
        audit.assert_not_called()

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
        self.assertEqual(difference, {(1, 0): (0, 0)})
        self.assertEqual((cis, difference), own_voice_bench.intervals([first, first], replicates=100))

    def test_default_report_is_byte_identical_to_pre_parakeet_output(self):
        scores = [evaluate_sample("a", "b"), evaluate_sample("a b c", "a b c")]
        measured = [self.measurement(scores), self.measurement(scores)]
        for result, label in zip(measured, bench.BACKENDS, strict=True):
            result.backend = label
        report = own_voice_bench.own_report(measured, "en-US")
        # Captured from the original runner, not recomputed from a second new path.
        self.assertEqual(hashlib.sha256(report.encode()).hexdigest(),
                         "94cb86dc4cf1fdd377261d71167803ac8bf7d2c2ca244bcfac734d01635cf13d")

    def test_three_backend_resampling_keeps_candidate_and_baseline_paired(self):
        # Each clip improves by one error: paired differences are always -0.5,
        # while independent resampling would produce a much wider interval.
        apple = self.measurement([evaluate_sample("a b", "x y"), evaluate_sample("a b", "x y")])
        whisper = self.measurement([evaluate_sample("a b", "x y"), evaluate_sample("a b", "a x")])
        parakeet = self.measurement([evaluate_sample("a b", "a x"), evaluate_sample("a b", "a b")])
        cis, differences = own_voice_bench.intervals([apple, whisper, parakeet], replicates=100)
        self.assertEqual(len(cis), 3)
        self.assertEqual(cis[2][0], (0, 0.5))
        self.assertEqual(differences[(2, 1)], (-0.5, -0.5))
        self.assertEqual(differences[(1, 0)], (-0.5, 0))
        fourth = self.measurement([evaluate_sample("a b", "a b"), evaluate_sample("a b", "a b")])
        more_cis, more_differences = own_voice_bench.intervals([apple, whisper, parakeet, fourth], replicates=100)
        self.assertEqual(len(more_cis), 4)
        self.assertEqual(more_differences[(3, 1)], (-1, -0.5))
        measured = [apple, whisper, parakeet, fourth]
        for row, label in zip(measured, (*bench.BACKENDS, bench.PARAKEET_BACKEND, "fourth"), strict=True):
            row.backend = label
        self.assertIn("Paired WER difference (fourth minus Whisper)", own_voice_bench.own_report(measured, "en-US"))

    def test_unpaired_or_empty_bootstrap_is_refused(self):
        first = self.measurement([evaluate_sample("a", "a"), evaluate_sample("b", "b")])
        second = self.measurement([])
        for measured in ([], [first], [first, second]):
            with self.subTest(count=len(measured)), self.assertRaises(ValueError):
                own_voice_bench.intervals(measured)
        with self.assertRaises(ValueError):
            own_voice_bench.intervals([first, first], replicates=1)
        first.latencies.pop()
        with self.assertRaises(ValueError):
            own_voice_bench.intervals([first, first])


class ParakeetAdapterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="scrivo-parakeet-bench-test-")
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        (self.folder / "config.json").write_text("{}")
        (self.folder / "model.safetensors").touch()

    def test_snapshot_uses_only_requested_cache_and_exact_pin(self):
        cache = self.folder / "cache"
        download = MagicMock(return_value=str(self.folder))
        with patch.dict(sys.modules, huggingface_hub=SimpleNamespace(snapshot_download=download)), \
             patch.dict(os.environ):
            self.assertEqual(bench.parakeet_model_path(cache), str(self.folder))
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["HF_HUB_DISABLE_TELEMETRY"], "1")
        download.assert_called_once_with(bench.MODEL_PROFILES["parakeet"].repo,
                                         revision=bench.MODEL_REVISIONS[bench.PARAKEET_REPO],
                                         cache_dir=str(cache.resolve()), local_files_only=True)

    def test_missing_or_incomplete_cache_fails_without_download_retry(self):
        for missing in ("config.json", "model.safetensors", "snapshot"):
            with self.subTest(missing=missing):
                if missing != "snapshot":
                    (self.folder / missing).unlink()
                download = MagicMock(return_value=str(self.folder))
                if missing == "snapshot":
                    download.side_effect = FileNotFoundError(MARKER)
                with patch.dict(sys.modules, huggingface_hub=SimpleNamespace(snapshot_download=download)), \
                     patch.dict(os.environ), self.assertRaises(bench.BenchError) as raised:
                    bench.parakeet_model_path(self.folder / "cache")
                self.assertIn("pinned Parakeet model is not cached", str(raised.exception))
                self.assertNotIn(MARKER, str(raised.exception))
                download.assert_called_once()
                self.assertTrue(download.call_args.kwargs["local_files_only"])
                if missing != "snapshot":
                    (self.folder / missing).write_text("{}")

    def test_local_load_float_array_decode_and_model_reuse(self):
        model, parameter, core = MagicMock(), MagicMock(), MagicMock()
        core.array.side_effect = lambda value: value
        model.generate.return_value = [SimpleNamespace(text="  decoded text  ")]
        from_config = MagicMock(return_value=model)
        logmel = MagicMock(return_value=object())
        flatten = MagicMock(return_value=[("weight", parameter)])
        unflatten = MagicMock(return_value=object())
        modules = {
            "mlx": SimpleNamespace(core=core), "mlx.core": core,
            "mlx.utils": SimpleNamespace(tree_flatten=flatten, tree_unflatten=unflatten),
            "parakeet_mlx": SimpleNamespace(),
            "parakeet_mlx.utils": SimpleNamespace(from_config=from_config),
            "parakeet_mlx.audio": SimpleNamespace(get_logmel=logmel),
        }
        with patch.dict(sys.modules, modules), patch.object(bench.wave, "open") as open_wav:
            # In-memory PCM boundary values only; no recording or audio file.
            wav = open_wav.return_value.__enter__.return_value
            wav.getnframes.return_value = 1600
            wav.readframes.return_value = b"\x00\x80\x00\x40" * 800
            backend = bench.Parakeet(self.folder)
            from_config.assert_not_called()
            for _ in range(2):
                self.assertEqual(backend.transcribe(Path("placeholder.wav")), "decoded text")
            wav.readframes.assert_called_with(1600)
        from_config.assert_called_once_with({})
        model.load_weights.assert_called_once_with(str(self.folder / "model.safetensors"))
        flatten.assert_called_once_with(model.parameters.return_value)
        parameter.astype.assert_called_once_with(core.bfloat16)
        unflatten.assert_called_once_with([("weight", parameter.astype.return_value)])
        model.update.assert_called_once_with(unflatten.return_value)
        audio, config = logmel.call_args.args
        self.assertEqual(audio.dtype, np.float32)
        np.testing.assert_array_equal(audio, np.tile([-1., .5], 800))
        self.assertIs(config, model.preprocessor_config)
        self.assertEqual(model.generate.call_count, 2)
        model.generate.assert_called_with(logmel.return_value)


class ConsolePrivacyTests(unittest.TestCase):
    def test_python_and_native_backend_output_are_discarded_in_subprocess(self):
        done = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_platform_baseline.py",
             "-k", "test_exact_schema_same_clips_and_no_private_output"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn(MARKER, done.stdout + done.stderr)

    def test_argument_errors_do_not_echo_private_values(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), self.assertRaises(SystemExit):
            own_voice_bench.main([MARKER, "--out", MARKER, "--platform-out", MARKER, "--unknown", MARKER])
        self.assertNotIn(MARKER, out.getvalue() + err.getvalue())
