"""Aggregate-only benchmark plumbing; no app state, History, or model downloads."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
from datetime import date
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import time
import wave

import benchmark
from evaluation import aggregate_results, evaluate_sample, required_term_accuracy, tokenize
from scripts.apple_speech_backend import AppleSpeech, SpeechUnavailable
from speech_config import MODEL_PROFILES, MODEL_REVISIONS

ROOT = Path(__file__).resolve().parents[1]
WHISPER_REPO = MODEL_PROFILES["auto"].repo
WHISPER_REVISION = MODEL_REVISIONS[WHISPER_REPO]
BACKENDS = ("apple-speech-on-device", f"mlx-whisper-{WHISPER_REPO}")
PLATFORM_COLUMNS = ("backend", "n", "wer", "required_term_hit_rate", "median_latency_s", "date", "macos_version")
SPEECH_ERRORS = {
    "main-thread": "Apple Speech requires the main thread.",
    "locale": "Apple Speech does not support the requested locale.",
    "on-device": "Apple Speech has no on-device support for this locale; no server fallback was attempted.",
    "unavailable": "Apple Speech is unavailable; no server fallback was attempted.",
    "purpose-string": "Speech permission needs a host purpose string; see own-voice-bench.md.",
    "permission": "Speech Recognition permission was not granted. Allow it manually before running again.",
    "timeout": "Apple Speech timed out; no report was written.",
    "recognition": "Apple Speech failed; private error details were suppressed. No report was written.",
}


class BenchError(RuntimeError):
    pass


class PrivateParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's normal diagnostic echoes bad arguments, which can be private.
        self.exit(2, "Invalid arguments. Use --help for the supported options.\n")


@contextmanager
def quiet_backends():
    """Discard Python and native stdout/stderr, including ffmpeg child output.

    This process-wide guard is for these single-threaded CLI runners only. No
    captured output is kept on disk or replayed when a backend raises.
    """
    with open(os.devnull, "w") as sink:
        sys.stdout.flush()
        sys.stderr.flush()
        saved = [os.dup(fd) for fd in (1, 2)]
        try:
            for fd in (1, 2):
                os.dup2(sink.fileno(), fd)
            with redirect_stdout(sink), redirect_stderr(sink):
                try:
                    yield
                finally:
                    # Flush libc as well as Python before restoring descriptors.
                    if os.name == "posix":
                        import ctypes

                        ctypes.CDLL(None).fflush(None)
                    sink.flush()
        finally:
            for fd, copy in zip((1, 2), saved, strict=True):
                os.dup2(copy, fd)
                os.close(copy)


def outside_repo(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise BenchError("Keep recordings, references and model caches outside the repository.")
    return resolved


def load_rows(manifest, locale):
    if not re.fullmatch(r"[a-z]{2,3}-[A-Z]{2}", locale):
        raise BenchError("Use a language-region locale, for example en-US.")
    try:
        rows = benchmark.load_manifest(outside_repo(Path(manifest)))
        if not 10 <= len(rows) <= 30:
            raise BenchError("A run requires 10 to 30 clips; no clips were measured.")
        if len({row.sample_id for row in rows}) != len(rows):
            raise BenchError("Manifest IDs must be unique.")
        for row in rows:
            outside_repo(row.audio)
            if not tokenize(row.reference):
                raise BenchError("Complete every exact reference before running.")
            if row.language != locale.split("-")[0]:
                raise BenchError("Every clip language must match the selected locale's language.")
            score, _, _ = required_term_accuracy(row.reference, row.required_terms)
            if score is not None and score != 1:
                raise BenchError("Required terms must occur in the reference.")
            with wave.open(str(row.audio), "rb") as wav:
                format_info = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype())
                if format_info != (1, 2, 16000, "NONE"):
                    raise BenchError("Audio must be uncompressed 16 kHz mono 16-bit WAV.")
                if not 0 < wav.getnframes() <= 30 * 16000:
                    raise BenchError("Each clip must be longer than zero and at most 30 seconds.")
        return rows
    except BenchError:
        raise
    except Exception:
        raise BenchError("Cannot validate the private manifest or WAV files. No report was written.") from None


def model_path(cache_dir):
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    from huggingface_hub import snapshot_download

    try:
        folder = Path(snapshot_download(WHISPER_REPO, revision=WHISPER_REVISION,
                                       cache_dir=str(outside_repo(Path(cache_dir))), local_files_only=True))
        if (not (folder / "config.json").is_file()
                or not any((folder / name).is_file() for name in ("weights.safetensors", "weights.npz"))):
            raise ValueError
        return str(folder)
    except Exception:
        raise BenchError("The pinned Whisper model is not cached. Follow model setup in own-voice-bench.md.") from None


@dataclass
class Measurement:
    backend: str
    overall: dict
    samples: list
    latencies: list[float]


def measure(rows, *, locale, cache_dir):
    """Same validated rows, sequential backends, no prompts or personal settings."""
    with quiet_backends():
        local_model = model_path(cache_dir)
        apple = AppleSpeech(locale)
        apple.authorize()
        results = []
        for label in BACKENDS:
            scores, latencies = [], []
            for row in rows:
                started = time.perf_counter()
                if label == BACKENDS[0]:
                    hypothesis = apple.transcribe(row.audio)
                else:
                    hypothesis = benchmark._transcribe(row.audio, local_model, row.language, None)
                elapsed = time.perf_counter() - started
                if not isinstance(hypothesis, str):
                    raise BenchError("A backend returned no transcript. No report was written.")
                score = evaluate_sample(row.reference, hypothesis, required_terms=row.required_terms)
                # Keep numeric sufficient statistics only after scoring, even in
                # memory. Never retain arbitrary language/tag slices in reports.
                scores.append(replace(score, reference="", hypothesis="", normalized_reference="",
                                      normalized_hypothesis=""))
                latencies.append(elapsed)
            results.append(Measurement(label, aggregate_results(scores)["overall"], scores, latencies))
        return results


def number(value):
    return "n/a" if value is None else f"{value:.6f}"


def table(columns, rows):
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |",
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)]) + "\n"


def platform_report(measurements):
    version = platform.mac_ver()[0]
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", version):
        raise BenchError("This benchmark requires macOS.")
    rows = []
    for result in measurements:
        metrics = result.overall["metrics"]
        rows.append((result.backend, result.overall["count"], number(metrics["wer"]["value"]),
                     number(metrics["required_term_accuracy"]["value"]), number(statistics.median(result.latencies)),
                     date.today().isoformat(), version))
    return table(PLATFORM_COLUMNS, rows)


def check_outputs(paths):
    targets = [Path(path).expanduser().absolute() for path in paths]
    if len({path.resolve() for path in targets}) != len(targets):
        raise BenchError("Reports need distinct output files.")
    if any(path.exists() or path.is_symlink() or path.suffix != ".md" for path in targets):
        raise BenchError("Choose new .md output files; existing files are never overwritten.")
    return targets


def write_reports(reports):
    # All computation succeeds before creating any output. Exclusive creation
    # also prevents replacing an input through a symlink or a concurrent writer.
    written = []
    try:
        for path, payload in reports:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as handle:
                written.append(path)
                handle.write(payload)
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise BenchError("Could not write the aggregate reports; no previous files were changed.") from None


def report_error(exc):
    if isinstance(exc, SpeechUnavailable):
        message = SPEECH_ERRORS.get(str(exc), "Apple Speech failed; private details were suppressed.")
    elif isinstance(exc, BenchError):
        message = str(exc)
    elif isinstance(exc, KeyboardInterrupt):
        message = "Benchmark cancelled; no complete measurement was reported."
    else:
        message = "Benchmark failed; private error details were suppressed. No complete measurement was reported."
    print(message, file=sys.stderr)
    return 1


def parser(description):
    result = PrivateParser(description=description)
    result.add_argument("manifest", metavar="MANIFEST")
    result.add_argument("--locale", default="en-US", metavar="LOCALE")
    result.add_argument("--cache-dir", metavar="CACHE", help="Private model cache; default: beside manifest/models")
    result.add_argument("--dry-run", action="store_true", help="Validate only; no recognition or report")
    return result
