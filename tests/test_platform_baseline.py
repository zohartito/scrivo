"""Stub-only privacy and Apple API contract checks; no audio or model is used."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from scripts import bench_common as bench, platform_baseline
from scripts.apple_speech_backend import AppleSpeech, SpeechUnavailable

MARKER = "PRIVATE-MARKER"


class BenchFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="scrivo-bench-test-")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.manifest = self.folder / f"{MARKER}.jsonl"
        self.output = self.folder / "platform.md"
        # Empty placeholders only. WAV metadata and recognizers are stubbed;
        # the tests neither synthesize nor decode any audio.
        self.entries = []
        for index in range(10):
            audio = self.folder / f"{MARKER}-{index}.wav"
            audio.touch()
            self.entries.append({"id": f"{MARKER}-{index}", "audio": audio.name,
                                 "reference": f"{MARKER} term", "language": "en",
                                 "tags": [MARKER], "required_terms": ["term"]})
        self.save_manifest()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.wav = self.stack.enter_context(patch.object(bench.wave, "open")).return_value.__enter__.return_value
        for name, value in (("getnchannels", 1), ("getsampwidth", 2), ("getframerate", 16000),
                            ("getcomptype", "NONE"), ("getnframes", 16000)):
            getattr(self.wav, name).return_value = value
        self.stack.enter_context(patch.object(platform_baseline, "ensure_host"))
        self.stack.enter_context(patch.object(bench, "model_path", return_value="/private-model"))
        self.stack.enter_context(patch.object(bench.platform, "mac_ver", return_value=("27.2", (), "")))
        self.apple_class = self.stack.enter_context(patch.object(bench, "AppleSpeech"))
        self.apple = self.apple_class.return_value
        self.apple.transcribe.side_effect = self.noisy_transcript
        self.whisper = self.stack.enter_context(patch.object(bench.benchmark, "_transcribe",
                                                           side_effect=self.noisy_transcript))

    def save_manifest(self):
        self.manifest.write_text("\n".join(json.dumps(row) for row in self.entries), encoding="utf-8")

    @staticmethod
    def noisy_transcript(*args):
        print(MARKER)
        print(MARKER, file=__import__("sys").stderr)
        os.write(1, MARKER.encode())
        os.write(2, MARKER.encode())
        return f"{MARKER} term"

    def invoke(self, extra=(), main=platform_baseline.main):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main([str(self.manifest), "--out", str(self.output), *extra])
        self.assertNotIn(MARKER, out.getvalue() + err.getvalue())
        return code, out.getvalue(), err.getvalue()


class PlatformBaselineTests(BenchFixture):
    def test_exact_schema_same_clips_and_no_private_output(self):
        code, out, err = self.invoke()
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        self.assertEqual(self.output.read_text(), out)
        self.assertEqual(tuple(item.strip() for item in out.splitlines()[0].strip("|").split("|")),
                         bench.PLATFORM_COLUMNS)
        self.assertEqual(len(out.splitlines()), 4)
        self.assertEqual(self.apple.transcribe.call_count, 10)
        self.assertEqual(self.whisper.call_count, 10)
        self.assertEqual([call.args[0] for call in self.apple.transcribe.call_args_list],
                         [call.args[0] for call in self.whisper.call_args_list])
        for call in self.whisper.call_args_list:
            self.assertEqual(call.args[1:], ("/private-model", "en", None))

    def test_dry_run_never_loads_backends_or_writes(self):
        self.assertEqual(self.invoke(["--dry-run"])[0], 0)
        self.apple_class.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_rejects_incomplete_and_inconsistent_manifests(self):
        original = list(self.entries)
        for change in ("short", "long", "blank", "duplicate", "language", "term"):
            with self.subTest(change=change):
                self.entries = [dict(row) for row in original]
                if change == "short": self.entries.pop()
                if change == "long": self.entries *= 4
                if change == "blank": self.entries[0]["reference"] = ""
                if change == "duplicate": self.entries[0]["id"] = self.entries[1]["id"]
                if change == "language": self.entries[0]["language"] = "fr"
                if change == "term": self.entries[0]["required_terms"] = ["absent"]
                self.save_manifest()
                self.assertEqual(self.invoke()[0], 1)
                self.assertFalse(self.output.exists())
        self.apple_class.assert_not_called()

    def test_decode_failure_is_redacted_without_partial_report(self):
        self.whisper.side_effect = RuntimeError(MARKER)
        code, out, err = self.invoke()
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("suppressed", err)
        self.assertFalse(self.output.exists())

    def test_unsupported_on_device_stops_before_whisper(self):
        self.apple_class.side_effect = SpeechUnavailable("on-device")
        self.assertIn("no on-device support", self.invoke()[2])
        self.whisper.assert_not_called()

    def test_bad_manifest_and_argument_diagnostics_are_redacted(self):
        self.manifest.write_text(MARKER)
        self.assertEqual(self.invoke()[0], 1)
        with self.assertRaises(SystemExit) as raised:
            self.invoke(["--unknown", MARKER])
        self.assertEqual(raised.exception.code, 2)

    def test_existing_report_and_bad_wav_are_refused(self):
        self.output.write_text("existing")
        self.assertEqual(self.invoke()[0], 1)
        self.assertEqual(self.output.read_text(), "existing")
        self.output.unlink()
        self.wav.getnchannels.return_value = 2
        self.assertEqual(self.invoke()[0], 1)
        self.apple_class.assert_not_called()

    def test_statistics_retained_in_memory_have_no_text_or_identifiers(self):
        rows = bench.load_rows(self.manifest, "en-US")
        measured = bench.measure(rows, locale="en-US", cache_dir=self.folder / "models")
        self.assertNotIn(MARKER, repr(measured))
        self.assertEqual(measured[0].overall["metrics"]["wer"]["value"], 0)


class AppleSpeechTests(unittest.TestCase):
    def setUp(self):
        self.speech, self.foundation = MagicMock(), MagicMock()
        self.foundation.NSThread.isMainThread.return_value = True
        self.speech.SFSpeechRecognizerAuthorizationStatusNotDetermined = 0
        self.speech.SFSpeechRecognizerAuthorizationStatusAuthorized = 3
        self.cls = self.speech.SFSpeechRecognizer
        self.cls.authorizationStatus.return_value = 3
        locale = MagicMock()
        locale.localeIdentifier.return_value = "en_US"
        self.cls.supportedLocales.return_value = [locale]
        self.recognizer = self.cls.alloc.return_value.initWithLocale_.return_value
        self.recognizer.supportsOnDeviceRecognition.return_value = True
        self.recognizer.isAvailable.return_value = True
        self.backend = AppleSpeech("en-US", speech=self.speech, foundation=self.foundation, timeout=0)

    def test_on_device_only_final_result_and_task_cleanup(self):
        partial, final = MagicMock(), MagicMock()
        partial.isFinal.return_value = False
        final.isFinal.return_value = True
        final.bestTranscription.return_value.formattedString.return_value = MARKER
        task = MagicMock()

        def start(request, callback):
            callback(partial, None)
            callback(final, None)
            return task

        self.recognizer.recognitionTaskWithRequest_resultHandler_.side_effect = start
        self.assertEqual(self.backend.transcribe(Path(MARKER)), MARKER)
        request = self.speech.SFSpeechURLRecognitionRequest.alloc.return_value.initWithURL_.return_value
        request.setRequiresOnDeviceRecognition_.assert_called_once_with(True)
        request.setShouldReportPartialResults_.assert_called_once_with(False)
        task.cancel.assert_called_once()

    def test_timeout_cancels_and_does_not_retry(self):
        with self.assertRaisesRegex(SpeechUnavailable, "timeout"):
            self.backend.transcribe(Path(MARKER))
        self.recognizer.recognitionTaskWithRequest_resultHandler_.return_value.cancel.assert_called_once()
        self.recognizer.recognitionTaskWithRequest_resultHandler_.assert_called_once()

    def test_error_and_callback_exception_do_not_expose_details(self):
        for result, error in ((None, MARKER), (SimpleNamespace(isFinal=lambda: 1 / 0), None)):
            def start(request, callback, result=result, error=error):
                callback(result, error)
                return MagicMock()
            self.recognizer.recognitionTaskWithRequest_resultHandler_.side_effect = start
            with self.assertRaisesRegex(SpeechUnavailable, "^recognition$"):
                self.backend.transcribe(Path(MARKER))

    def test_denied_missing_purpose_and_async_authorization(self):
        self.cls.authorizationStatus.return_value = 2
        with self.assertRaisesRegex(SpeechUnavailable, "permission"):
            self.backend.authorize()
        self.cls.authorizationStatus.return_value = 0
        info = self.foundation.NSBundle.mainBundle.return_value
        info.infoDictionary.return_value = {}
        with self.assertRaisesRegex(SpeechUnavailable, "purpose-string"):
            self.backend.authorize()
        self.cls.requestAuthorization_.assert_not_called()
        info.infoDictionary.return_value = {"NSSpeechRecognitionUsageDescription": "benchmark"}
        self.cls.requestAuthorization_.side_effect = lambda callback: callback(3)
        self.backend.authorize()
        self.cls.requestAuthorization_.assert_called_once()

    def test_no_locale_fallback_and_no_server_fallback(self):
        with self.assertRaisesRegex(SpeechUnavailable, "locale"):
            AppleSpeech("fr-FR", speech=self.speech, foundation=self.foundation)
        self.recognizer.supportsOnDeviceRecognition.return_value = False
        with self.assertRaisesRegex(SpeechUnavailable, "on-device"):
            self.backend.transcribe(Path(MARKER))
        self.recognizer.recognitionTaskWithRequest_resultHandler_.assert_not_called()
