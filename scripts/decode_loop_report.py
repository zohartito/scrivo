"""Where pasted repetition loops come from: greedy (T=0) decodes or fallbacks.

Reads History's ``history.jsonl`` read-only with a plain JSON-lines parse
(never HistoryStore, whose constructor sweeps orphan audio) and prints counts
only: never a transcript, a repeated word or an audio path. A loop is a live
dictation whose loop was cut (``repetition_trimmed``) or that was held back as
hallucinated (pipeline.looks_hallucinated on the kept text). Its decode comes
from the row's ``preprocessing.decode`` block (speech_config.full_decode_summary);
rows from before that block count as "no telemetry", Nemotron and Parakeet
rows as "other engine".

Run from the repo root:
    venv-alpha/bin/python scripts/decode_loop_report.py --days 7 [--data-dir PATH]
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import looks_hallucinated, reads_as_no_speech  # noqa: E402

# Scrivo.app's data folder (its sotto-alpha name was kept on purpose).
DEFAULT_DATA_DIR = Path.home() / "Library/Application Support/sotto-alpha"
WHISPER_PATHS = ("full", "one_pass")
OTHER_ENGINE_PROFILES = ("nemotron-en", "parakeet")
TEMPERATURE_BUCKETS = ("0.0", "0.2-0.4", "0.6+", "n/a")
COMPRESSION_BUCKETS = ("<= 2.4", "2.4-3.0", "> 3.0", "n/a")
# Scrivo's own placeholder rows ("[no speech detected]", "[dead microphone]",
# "[transcription failed]", "[not transcribed: Scrivo stopped]"): no transcript.
MARKER_TEXT = re.compile(r"\[[^\[\]]*\]")
ERROR_KEYS = ("error", "transcription_error")  # Mac, Windows


def read_rows(index: Path) -> tuple[list[dict], int]:
    """Every JSON object in ``index`` and how many lines were not one (a line
    the app is still writing, or damage). Opens the file for reading only."""
    rows, unreadable = [], 0
    with index.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                unreadable += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                unreadable += 1
    return rows, unreadable


def live_attempt(row: dict) -> dict:
    """The live dictation's own attempt (a Retry or correction changes the
    row's text, never its initial attempt); the row itself for old layouts."""
    attempts = row.get("attempts")
    if isinstance(attempts, list) and attempts and isinstance(attempts[0], dict):
        return attempts[0]
    return row


def _number(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def screened_seconds(row: dict, attempt: dict) -> float | None:
    """The clip length pipeline.screen judged: the canonical audio's exact
    sample count (History rounds ``duration`` to 0.01 s), else ``duration``."""
    audio = row.get("audio") if isinstance(row.get("audio"), dict) else {}
    inference = audio.get("inference") if isinstance(audio.get("inference"), dict) else {}
    count, rate = _number(inference.get("sample_count")), _number(inference.get("sample_rate"))
    if count is not None and rate:
        return count / rate
    return _number(attempt.get("duration"))


def loop_kind(attempt: dict, seconds: float | None) -> str | None:
    """"salvaged", "flagged" or None, by pipeline.screen's own rules."""
    preprocessing = attempt.get("preprocessing")
    if not isinstance(preprocessing, dict):
        return None
    if "repetition_trimmed" in preprocessing:
        return "salvaged"
    if preprocessing.get("outcome") != "suspect" or any(key in preprocessing for key in ERROR_KEYS):
        return None  # delivered, or a capture whose model never produced text
    text = attempt.get("text")
    if not isinstance(text, str) or seconds is None or MARKER_TEXT.fullmatch(text):
        return None
    vad = attempt.get("vad") if isinstance(attempt.get("vad"), dict) else {}
    fraction = _number(vad.get("speech_fraction"))
    if reads_as_no_speech(text, 1.0 if fraction is None else fraction):
        return None  # held as silence, not as a loop
    return "flagged" if looks_hallucinated(text, seconds) else None


def telemetry(attempt: dict) -> tuple[str, dict | None]:
    """("whisper", decode block) | ("other engine", None) | ("no telemetry", None)."""
    preprocessing = attempt.get("preprocessing") if isinstance(attempt.get("preprocessing"), dict) else {}
    decode = preprocessing.get("decode")
    if isinstance(decode, dict):
        return ("whisper", decode) if decode.get("path") in WHISPER_PATHS else ("other engine", None)
    if attempt.get("profile") in OTHER_ENGINE_PROFILES:
        return "other engine", None
    return "no telemetry", None


def decode_route(decode: dict) -> tuple[str, float | None, float | None, float | None]:
    """(path, max temperature, max compression ratio, max no-speech prob) of
    the decode whose text was kept."""
    full = decode.get("full")
    if decode.get("path") == "one_pass" and not isinstance(full, dict):
        return ("one_pass", 0.0, _number(decode.get("compression_ratio")),
                _number(decode.get("no_speech_prob")))
    if not isinstance(full, dict):
        full = decode
    path = "one_pass -> full" if decode.get("path") == "one_pass" else "full"
    return (path, _number(full.get("max_temperature")), _number(full.get("max_compression_ratio")),
            _number(full.get("max_no_speech_prob")))


def temperature_bucket(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value == 0:
        return "0.0"
    return "0.2-0.4" if value <= 0.4 else "0.6+"


def compression_bucket(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value <= 2.4:
        return "<= 2.4"
    return "2.4-3.0" if value <= 3.0 else "> 3.0"


def oldest(rows: list[dict]) -> float | None:
    """When History's oldest row was dictated (it keeps only its latest rows)."""
    return min((ts for ts in (_number(row.get("ts")) for row in rows) if ts is not None), default=None)


def summarize(rows: list[dict], since: float) -> Counter:
    """Counts only. Keys are (section, label) pairs."""
    counts: Counter = Counter()
    for row in rows:
        ts = _number(row.get("ts"))
        if ts is None or ts < since:
            continue
        attempt = live_attempt(row)
        source, decode = telemetry(attempt)
        counts["dictations", "all"] += 1
        counts["dictations", source] += 1
        if decode is not None:
            path, temperature, _ratio, _no_speech = decode_route(decode)
            counts["whisper path", path] += 1
            counts["whisper temperature", temperature_bucket(temperature)] += 1
        kind = loop_kind(attempt, screened_seconds(row, attempt))
        if kind is None:
            continue
        counts["loops", "all"] += 1
        counts["loops", kind] += 1
        counts["loops", source] += 1
        if decode is not None:
            path, temperature, ratio, no_speech = decode_route(decode)
            counts["loop path", path] += 1
            counts["loop temperature", temperature_bucket(temperature)] += 1
            counts["loop compression", compression_bucket(ratio)] += 1
            if no_speech is not None and no_speech > 0.6:
                counts["loop", "no-speech prob > 0.6"] += 1
    return counts


LABELS = {"whisper": "Whisper, with decode telemetry", "salvaged": "salvaged (loop cut, prefix pasted)",
          "flagged": "flagged (held back, not pasted)"}
PATHS = ("full", "one_pass", "one_pass -> full")


def _when(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def render(counts: Counter, *, days: int, since: float, first: float | None, unreadable: int) -> str:
    def total(name: str, count: int) -> str:
        return f"{name:<42}{count:>6}"

    def block(section: str, labels) -> list[str]:
        return [f"    {LABELS.get(label, label):<38}{counts[section, label]:>6}"
                for label in labels if counts[section, label]]

    lines = [f"Decode loop report: last {days} day(s), since {_when(since)}"]
    if first is not None and first > since:
        lines.append(f"(History keeps only its latest rows; the oldest is from {_when(first)})")
    lines += ["", total("Dictations", counts["dictations", "all"]),
              *block("dictations", ("whisper", "no telemetry", "other engine"))]
    if counts["dictations", "whisper"]:
        lines += ["  Whisper decode path", *block("whisper path", PATHS),
                  "  Whisper max temperature", *block("whisper temperature", TEMPERATURE_BUCKETS)]
    lines += ["", total("Repetition loops", counts["loops", "all"]),
              *block("loops", ("salvaged", "flagged"))]
    if counts["loops", "all"]:
        lines += ["  Telemetry", *block("loops", ("whisper", "no telemetry", "other engine"))]
    if counts["loops", "whisper"]:
        lines += ["  Loop decode path", *block("loop path", PATHS),
                  "  Loop max temperature (0.0 = greedy)", *block("loop temperature", TEMPERATURE_BUCKETS),
                  "  Loop max compression ratio (fallback above 2.4)", *block("loop compression", COMPRESSION_BUCKETS),
                  "  Loop kept as silence despite that", *block("loop", ("no-speech prob > 0.6",))]
    if unreadable:
        lines += ["", total("Unreadable lines skipped", unreadable)]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=7, help="how many days back to count (default 7)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help=f"Scrivo's data folder (default {DEFAULT_DATA_DIR})")
    args = parser.parse_args(argv)
    if args.days < 1:
        parser.error("--days must be at least 1")
    index = args.data_dir.expanduser() / "history.jsonl"
    if not index.is_file():
        print(f"No History at {index}", file=sys.stderr)
        return 1
    rows, unreadable = read_rows(index)
    since = time.time() - args.days * 86_400
    print(render(summarize(rows, since), days=args.days, since=since, first=oldest(rows), unreadable=unreadable))
    return 0


if __name__ == "__main__":
    sys.exit(main())
