"""Decode telemetry: which Whisper decode step produced a transcript.

Scrivo never sets a temperature, so mlx-whisper's full pipeline uses its
default fallback schedule (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) while
LocalWhisper.one_pass is always one greedy (T=0) decode. The "decode" block
records which step a transcript came from, so a pasted repetition loop
("difference" x223, 2026-08-15) can be traced to T=0 or to a fallback. It is
recorded only: these tests also pin that the returned text never changes.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from history import HistoryStore
import scrivo
from pipeline import move_detected_language, screen
from speech_config import MODEL_PROFILES, resolve_speech_config

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the golden master's harness, under any runner
from test_transcription_worker import LOOP, PREFIX, Harness, TemporaryUserFiles, says, speech  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("decode_loop_report", ROOT / "scripts" / "decode_loop_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)

LOOP_PREFIX = "This part of the dictation is real and should be kept intact."
SAMPLES = np.zeros(16_000, dtype=np.float32)  # one second: a short clip


def segment(temperature, compression_ratio, avg_logprob=-0.2, no_speech_prob=0.01):
    """One mlx_whisper.transcribe segment with the decode fields it carries."""
    return {"text": "...", "temperature": temperature, "compression_ratio": compression_ratio,
            "avg_logprob": avg_logprob, "no_speech_prob": no_speech_prob}


class FakeBackend:
    """Stands in for the mlx_whisper module: returns one fixed result."""

    def __init__(self, result: dict):
        self.result = result
        self.calls = []

    def transcribe(self, samples, **kwargs):
        self.calls.append(kwargs)
        return self.result


def fake_decoding(decoded, options_seen=None):
    """mlx_whisper.decoding with a DecodingTask whose run() returns ``decoded``
    (never imports the real mlx_whisper into the test process). Each
    DecodingOptions one_pass builds is appended to ``options_seen``."""
    task = Mock(sample_len=224)
    task.run.return_value = [decoded]

    def options(**given):
        if options_seen is not None:
            options_seen.append(given)
        return given

    module = types.SimpleNamespace(DecodingOptions=options, DecodingTask=lambda *_a, **_k: task)
    return patch.dict(sys.modules, {"mlx_whisper": types.SimpleNamespace(decoding=module),
                                    "mlx_whisper.decoding": module})


class BadSegment(dict):
    """A segment whose accessor itself fails."""

    def get(self, *_args):
        raise ValueError("bad metric")


class BadResult:
    """A one_pass DecodingResult that ran out of tokens and whose metrics fail."""
    tokens = [1] * 224
    text = " cut off"

    @property
    def compression_ratio(self):
        raise ValueError("bad metric")


def greedy(text, *, compression_ratio, avg_logprob=-0.2, no_speech_prob=0.01, tokens=20):
    """A one_pass DecodingResult (no temperature attribute: one_pass fixes 0.0)."""
    return types.SimpleNamespace(tokens=[1] * tokens, text=text, compression_ratio=compression_ratio,
                                 avg_logprob=avg_logprob, no_speech_prob=no_speech_prob)


class DecodeTelemetryTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.model_dir = folder.name  # an existing folder: no snapshot resolution, no network

    def whisper(self, result: dict) -> scrivo.LocalWhisper:
        return scrivo.LocalWhisper(FakeBackend(result), self.model_dir)

    def transcribe(self, local, language="en"):
        config = resolve_speech_config("auto", language=language)
        return scrivo.transcribe_canonical_samples(local, SAMPLES, config, ())

    def assert_json_safe(self, metadata):
        json.dumps(metadata, allow_nan=False)  # raises on NaN/inf or a non-JSON value

    # (a) clean speech decoded at T=0
    def test_full_pipeline_greedy_decode(self):
        result = {"text": " Hello there. How are you?", "language": "en",
                  "segments": [segment(0.0, 1.4, -0.21, 0.02), segment(0.0, 1.6, -0.3, 0.01)]}
        text, metadata = self.transcribe(self.whisper(result))
        self.assertEqual(text, "Hello there. How are you?")
        self.assertEqual(text, result["text"].strip(), "the backend's text, unchanged")
        self.assertEqual(metadata["decode"], {
            "path": "full", "segments": 2, "temperatures": [0.0, 0.0], "compression_ratios": [1.4, 1.6],
            "max_temperature": 0.0, "max_compression_ratio": 1.6, "min_avg_logprob": -0.3,
            "max_no_speech_prob": 0.02, "fallback_used": False})
        self.assert_json_safe(metadata)

    # (b) a loop kept at T=0: its no-speech probability cancelled the fallback
    def test_full_pipeline_greedy_loop_reaches_the_row_with_its_salvage(self):
        looped = " " + LOOP_PREFIX + " difference" * 40
        result = {"text": looped, "language": "en", "segments": [segment(0.0, 3.1, -0.15, 0.72)]}
        text, metadata = self.transcribe(self.whisper(result))
        self.assertEqual(text, looped.strip(), "the backend's text, unchanged")
        decode = metadata["decode"]
        self.assertEqual((decode["temperatures"], decode["compression_ratios"]), ([0.0], [3.1]))
        self.assertEqual((decode["max_temperature"], decode["max_compression_ratio"], decode["fallback_used"]),
                         (0.0, 3.1, False))
        self.assertEqual(decode["max_no_speech_prob"], 0.72)
        # What History keeps: the block moves into preprocessing beside the salvage receipt.
        preprocessing = {}
        move_detected_language(metadata, preprocessing)
        self.assertNotIn("decode", metadata)
        screened = screen(text, seconds=3.0, speech_fraction=0.9, language="en", clean=False)
        preprocessing.update(screened.receipts)
        self.assertEqual(screened.text, LOOP_PREFIX)
        self.assertIs(preprocessing["decode"], decode)
        self.assertEqual(preprocessing["repetition_trimmed"]["unit"], "difference")

    # (c) a window that needed the T=0.4 fallback
    def test_full_pipeline_temperature_fallback(self):
        result = {"text": " First window. Second window.", "language": "en",
                  "segments": [segment(0.0, 1.3), segment(0.4, 2.0, -0.8, 0.1)]}
        text, metadata = self.transcribe(self.whisper(result))
        self.assertEqual(text, "First window. Second window.")
        self.assertEqual(text, result["text"].strip(), "the backend's text, unchanged")
        decode = metadata["decode"]
        self.assertEqual(decode["temperatures"], [0.0, 0.4])
        self.assertEqual(decode["compression_ratios"], [1.3, 2.0], "segment by segment with temperatures")
        self.assertEqual(decode["max_temperature"], 0.4)
        self.assertTrue(decode["fallback_used"])
        self.assertEqual(decode["min_avg_logprob"], -0.8)

    def test_temperatures_are_bounded(self):
        result = {"text": " long", "segments": [segment(0.2, 1.0)] * 50}
        _text, metadata = self.transcribe(self.whisper(result))
        self.assertEqual(metadata["decode"]["segments"], 50)
        self.assertEqual(len(metadata["decode"]["temperatures"]), 32)
        self.assertEqual(len(metadata["decode"]["compression_ratios"]), 32)

    def test_compression_ratios_stay_aligned_and_finite(self):
        result = {"text": " ok", "segments": [segment(0.0, 1.2), segment(0.2, float("nan")),
                                              segment(float("nan"), 2.9), segment(0.4, 2.6)]}
        _text, metadata = self.transcribe(self.whisper(result))
        self.assertEqual(metadata["decode"]["temperatures"], [0.0, 0.2, 0.4])
        self.assertEqual(metadata["decode"]["compression_ratios"], [1.2, None, 2.6])
        self.assert_json_safe(metadata)

    # (d) one_pass: accepted, and handed to the full pipeline
    def automatic(self, decoded, full_result, options_seen=None):
        local = self.whisper(full_result)
        local.allowed_languages = lambda: ("en", "pt")
        with patch.object(scrivo.LocalWhisper, "_model", return_value=Mock()), \
             patch.object(scrivo.LocalWhisper, "_features", return_value=Mock()), \
             patch.object(scrivo.LocalWhisper, "_language_probabilities", return_value={"en": 0.9, "pt": 0.1}), \
             fake_decoding(decoded, options_seen):
            text, metadata = self.transcribe(local, language="auto")
        return local, text, metadata

    def test_one_pass_accepted(self):
        options_seen = []
        local, text, metadata = self.automatic(greedy(" Hello.", compression_ratio=1.1), {"text": "unused"},
                                               options_seen)
        self.assertEqual(text, "Hello.")
        self.assertEqual(options_seen, [{"language": "en", "without_timestamps": True, "temperature": 0.0}],
                         "one_pass stays one greedy decode")
        self.assertEqual(local.backend.calls, [], "an accepted one_pass never runs the full pipeline")
        self.assertEqual(metadata["decode"], {
            "path": "one_pass", "temperature": 0.0, "compression_ratio": 1.1, "avg_logprob": -0.2,
            "no_speech_prob": 0.01, "outcome": "accepted"})
        self.assertEqual(metadata["detected_language"], "en")
        self.assert_json_safe(metadata)

    def test_one_pass_hand_off_records_both_decodes(self):
        full = {"text": " Hello hello.", "language": "en", "segments": [segment(0.0, 2.6), segment(0.2, 1.5)]}
        local, text, metadata = self.automatic(
            greedy(" hello" * 30, compression_ratio=3.1, no_speech_prob=0.3), full)
        self.assertEqual(text, "Hello hello.")
        self.assertEqual(local.backend.calls[0]["language"], "en")
        decode = metadata["decode"]
        self.assertEqual({key: decode[key] for key in ("path", "temperature", "compression_ratio", "outcome",
                                                       "reason")},
                         {"path": "one_pass", "temperature": 0.0, "compression_ratio": 3.1,
                          "outcome": "handed_off", "reason": "compression_ratio"})
        self.assertEqual(decode["full"]["temperatures"], [0.0, 0.2])
        self.assertTrue(decode["full"]["fallback_used"])
        self.assert_json_safe(metadata)

    def test_one_pass_other_outcomes(self):
        cases = [(greedy(" x", compression_ratio=1.0, avg_logprob=-1.4), "handed_off", "avg_logprob"),
                 (greedy(" x", compression_ratio=1.0, tokens=224), "handed_off", "token_budget"),
                 (greedy(" you", compression_ratio=1.0, avg_logprob=-1.2, no_speech_prob=0.9), "no_speech", None)]
        for decoded, outcome, reason in cases:
            with self.subTest(outcome=outcome, reason=reason):
                _local, _text, metadata = self.automatic(decoded, {"text": " full", "segments": []})
                self.assertEqual(metadata["decode"]["outcome"], outcome)
                self.assertEqual(metadata["decode"].get("reason"), reason)

    def test_a_block_never_leaks_into_the_next_transcription(self):
        full = {"text": " Again.", "segments": [segment(0.0, 1.2)]}
        local, _text, first = self.automatic(greedy(" hello" * 30, compression_ratio=3.1), full)
        _text, second = self.transcribe(local)  # a fixed language goes straight to the full pipeline
        self.assertEqual(first["decode"]["outcome"], "handed_off")
        self.assertEqual(set(second["decode"]) & {"outcome", "full"}, set())
        local.backend.transcribe = Mock(side_effect=RuntimeError("Metal"))
        with self.assertRaises(RuntimeError):
            self.transcribe(local)
        self.assertIsNone(local.last_decode, "a decode that raised leaves no block behind")

    # A malformed result never costs the dictation.
    def test_malformed_results_still_deliver_the_text(self):
        for segments in ("garbage", None, [None, "seg", {"temperature": float("nan"), "compression_ratio": "x",
                                                         "avg_logprob": None, "no_speech_prob": float("inf")}]):
            with self.subTest(segments=segments):
                text, metadata = self.transcribe(self.whisper({"text": " still here", "segments": segments}))
                self.assertEqual(text, "still here")
                self.assertEqual(metadata["decode"]["path"], "full")
                self.assertIsNone(metadata["decode"]["max_temperature"])
                self.assertFalse(metadata["decode"]["fallback_used"])
                self.assert_json_safe(metadata)

    def test_failing_accessors_and_numpy_scalars(self):
        text, metadata = self.transcribe(self.whisper({"text": " still here", "segments": [BadSegment()]}))
        self.assertEqual((text, metadata["decode"]), ("still here", {"path": "full", "unreadable": True}))
        numpy_segment = {"temperature": np.float32(0.2), "compression_ratio": np.float64(2.5),
                         "avg_logprob": np.float32(-0.5), "no_speech_prob": np.float64("nan")}
        _text, metadata = self.transcribe(self.whisper({"text": " ok", "segments": [numpy_segment]}))
        self.assertEqual((metadata["decode"]["temperatures"], metadata["decode"]["max_compression_ratio"],
                          metadata["decode"]["max_no_speech_prob"]), ([0.2], 2.5, None))
        self.assertIs(type(metadata["decode"]["temperatures"][0]), float)
        self.assert_json_safe(metadata)
        # one_pass: a budget-exhausted decode is still handed on, metrics or not
        local, text, metadata = self.automatic(BadResult(), {"text": " full text", "segments": []})
        self.assertEqual(text, "full text")
        self.assertEqual(len(local.backend.calls), 1)
        self.assertEqual({key: metadata["decode"][key] for key in ("unreadable", "outcome", "reason")},
                         {"unreadable": True, "outcome": "handed_off", "reason": "token_budget"})

    def test_decode_options_are_unchanged(self):
        local = self.whisper({"text": " ok", "segments": []})
        self.transcribe(local)
        self.assertEqual(set(local.backend.calls[0]),
                         {"path_or_hf_repo", "condition_on_previous_text", "word_timestamps", "language"})

    # Other engines: the engine name, no Whisper fields.
    def test_parakeet_and_nemotron_name_their_engine(self):
        with patch.object(scrivo, "resolve_pinned_snapshot", return_value=self.model_dir):
            parakeet = scrivo.LocalParakeet(MODEL_PROFILES["parakeet"].repo)
        parakeet.transcribe = lambda samples, **_kwargs: {"text": " ciao", "segments": [], "language": None}
        text, metadata = scrivo.transcribe_canonical_samples(
            parakeet, SAMPLES, resolve_speech_config("parakeet"), ())
        self.assertEqual((text, metadata["decode"]), ("ciao", {"path": "parakeet"}))

        nemotron = Mock()
        nemotron.transcribe.return_value = "hello"
        text, metadata = scrivo.transcribe_canonical_samples(
            None, SAMPLES, resolve_speech_config("nemotron-en"), (), nemotron=nemotron)
        self.assertEqual((text, metadata["decode"]), ("hello", {"path": "nemotron"}))

    @unittest.skipUnless(sys.platform == "win32", "win_asr imports only on Windows")
    def test_windows_backend_names_its_engine(self):
        import win_asr
        model = Mock()
        model.transcribe.return_value = {"text": " hola", "language": "es"}
        text, metadata = scrivo.transcribe_canonical_samples(
            win_asr.LanguageRestricted(model, ("en", "es")), SAMPLES, resolve_speech_config("auto"), ())
        self.assertEqual((text, metadata["decode"]), ("hola", {"path": "faster-whisper"}))


def full_block(temperatures, max_ratio, no_speech=0.01):
    return {"path": "full", "segments": len(temperatures), "temperatures": temperatures,
            "max_temperature": max(temperatures), "max_compression_ratio": max_ratio, "min_avg_logprob": -0.3,
            "max_no_speech_prob": no_speech, "fallback_used": max(temperatures) > 0}


class WorkerKeepsTheDecodeBlockTest(TemporaryUserFiles, unittest.TestCase):
    """The live worker (transcription.py) carries the block into the History row."""

    def test_salvaged_loop_row_keeps_its_decode_block(self):
        h = Harness(says(f"{PREFIX} {LOOP}"))
        h.whisper.last_decode = full_block([0.0], 3.1, no_speech=0.72)
        row = h.run_jobs(h.live_job(speech(12.0))).observed()["appends"][0]
        self.assertEqual(row["text"], PREFIX)
        self.assertEqual(row["preprocessing"]["decode"], full_block([0.0], 3.1, no_speech=0.72))
        self.assertEqual(row["preprocessing"]["repetition_trimmed"]["repeats"], 40)


class DecodeLoopReportTest(unittest.TestCase):
    """scripts/decode_loop_report.py over rows History itself wrote."""

    SECRET_LOOP = "ощ" * 223  # 2026-09-20's dead-mic loop; must never be printed

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.data = Path(folder.name)
        store = HistoryStore(self.data)
        now = time.time()
        voiced = {"available": True, "speech_fraction": 0.8, "span_count": 1}
        base = {"resampled_normalized": True, "vad_trimmed": False, "silence_collapsed": False}

        def add(text, seconds, profile="auto", provenance="live", days_ago=0.0, vad=voiced, samples=1600,
                **preprocessing):
            store.append(text, np.zeros(samples, np.float32), seconds, "model", ts=now - days_ago * 86_400,
                         profile=profile, language="en", vad=vad, provenance=provenance,
                         preprocessing={**base, **preprocessing})

        add("Clean dictation.", 2.0, decode=full_block([0.0], 1.4))
        add(LOOP_PREFIX, 3.0, decode=full_block([0.0], 3.1, no_speech=0.72),
            repetition_trimmed={"unit": "difference", "repeats": 40, "dropped_chars": 440})
        add(self.SECRET_LOOP, 5.0, provenance="live_suspect", outcome="suspect",
            vad={"available": True, "speech_fraction": 0.0, "span_count": 0},
            decode={"path": "one_pass", "temperature": 0.0, "compression_ratio": 2.9, "avg_logprob": -0.4,
                    "no_speech_prob": 0.3, "outcome": "handed_off", "reason": "compression_ratio",
                    "full": full_block([0.0, 0.2, 0.4, 0.6], 2.7)})
        add("Accepted greedy.", 1.5, decode={"path": "one_pass", "temperature": 0.0, "compression_ratio": 1.2,
                                              "avg_logprob": -0.2, "no_speech_prob": 0.02, "outcome": "accepted"})
        add("Nemotron words.", 2.0, profile="nemotron-en", decode={"path": "nemotron"})
        add("Nemotron before the block.", 2.0, profile="nemotron-en")
        add(LOOP_PREFIX, 3.0, repetition_trimmed={"unit": "difference", "repeats": 40, "dropped_chars": 440})
        add(scrivo.NO_SPEECH_TEXT, 0.5, provenance="live_suspect", outcome="suspect")
        add("[transcription failed]", 0.4, provenance="live_suspect", outcome="suspect", error="RuntimeError")
        add("you", 0.2, provenance="live_suspect", outcome="suspect",
            vad={"available": True, "speech_fraction": 0.0, "span_count": 0})
        # Windows keeps a failed capture under "transcription_error", not "error".
        add("[transcription failed]", 0.4, provenance="live_suspect", outcome="suspect",
            transcription_error="RuntimeError")
        add("[dead microphone]", 0.3, provenance="live_suspect", outcome="suspect")
        # 120 chars over 47,984 samples (2.999 s) is held (> 40 chars/s);
        # History's rounded duration (3.0 s) alone would read exactly 40.
        add("word " * 24, 47_984 / 16_000, samples=47_984, provenance="live_suspect", outcome="suspect")
        add("A month ago.", 2.0, days_ago=30, decode=full_block([0.0], 1.0))
        with (self.data / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write('{"id": "half-writ\n')
        self.before = {path: path.read_bytes() for path in self.data.rglob("*") if path.is_file()}

    def run_report(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = report.main(["--days", "7", "--data-dir", str(self.data), *args])
        return code, output.getvalue()

    def test_counts_loops_by_decode_route(self):
        rows, unreadable = report.read_rows(self.data / "history.jsonl")
        self.assertEqual(unreadable, 1)
        counts = report.summarize(rows, time.time() - 7 * 86_400)
        self.assertEqual(counts["dictations", "all"], 13)
        self.assertEqual((counts["dictations", "whisper"], counts["dictations", "no telemetry"],
                          counts["dictations", "other engine"]), (4, 7, 2))
        self.assertEqual((counts["loops", "all"], counts["loops", "salvaged"], counts["loops", "flagged"]), (4, 2, 2))
        self.assertEqual((counts["loops", "whisper"], counts["loops", "no telemetry"]), (2, 2))
        self.assertEqual(counts["loop path", "full"], 1)
        self.assertEqual(counts["loop path", "one_pass -> full"], 1)
        self.assertEqual((counts["loop temperature", "0.0"], counts["loop temperature", "0.6+"]), (1, 1))
        self.assertEqual((counts["loop compression", "> 3.0"], counts["loop compression", "2.4-3.0"]), (1, 1))
        self.assertEqual(counts["loop", "no-speech prob > 0.6"], 1)

    def test_classifies_from_exact_length_and_skips_placeholders(self):
        rows, _ = report.read_rows(self.data / "history.jsonl")
        boundary = next(row for row in rows if row.get("text") == "word " * 24)
        attempt = report.live_attempt(boundary)
        self.assertEqual(attempt["duration"], 3.0)
        self.assertIsNone(report.loop_kind(attempt, attempt["duration"]), "the rounded length alone misses it")
        self.assertEqual(report.loop_kind(attempt, report.screened_seconds(boundary, attempt)), "flagged")
        for row in rows:
            if str(row.get("text", "")).startswith("["):
                with self.subTest(text=row["text"], preprocessing=sorted(row["preprocessing"])):
                    attempt = report.live_attempt(row)
                    self.assertIsNone(report.loop_kind(attempt, report.screened_seconds(row, attempt)))

    def test_prints_counts_only_and_changes_nothing(self):
        code, output = self.run_report()
        self.assertEqual(code, 0)
        self.assertIn("Repetition loops", output)
        for private in ("difference", "ощ", LOOP_PREFIX, "Clean dictation", "audio", ".wav"):
            self.assertNotIn(private, output)
        after = {path: path.read_bytes() for path in self.data.rglob("*") if path.is_file()}
        self.assertEqual(after, self.before, "the report must leave History byte-for-byte untouched")

    def test_missing_history_is_an_error_not_a_new_folder(self):
        missing = self.data / "nowhere"
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            self.assertEqual(report.main(["--data-dir", str(missing)]), 1)
        self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
