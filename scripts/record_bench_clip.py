"""User-started private WAV capture. Never transcribes or supplies a reference."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sys
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import bench_common as bench  # noqa: E402
from scripts.bench_host import ensure_host  # noqa: E402


def read_entries(handle):
    handle.seek(0)
    try:
        entries = [json.loads(line) for line in handle.read().splitlines() if line.strip()]
        if any(not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in entries):
            raise ValueError
        if len({row["id"] for row in entries}) != len(entries):
            raise ValueError
        return entries
    except Exception:
        raise bench.BenchError("The existing manifest is invalid; it was left unchanged.") from None


def validate_id(clip_id):
    if not re.fullmatch(r"c(?:0[1-9]|[12][0-9]|30)", clip_id):
        raise bench.BenchError("Use a clip ID from c01 through c30.")


@contextmanager
def locked_manifest(folder):
    import fcntl

    folder = bench.outside_repo(Path(folder))
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Do not follow a pre-existing manifest symlink into another folder.
    fd = os.open(folder / "manifest.jsonl", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield handle


def append_entry(handle, clip_id, language):
    """Append a blank human-reference slot without rewriting prior entries."""
    validate_id(clip_id)
    if not re.fullmatch(r"[a-z]{2,3}", language):
        raise bench.BenchError("Use a two- or three-letter language code.")
    existing = read_entries(handle)
    if any(row["id"] == clip_id for row in existing):
        raise bench.BenchError("That clip ID already exists; choose a new ID.")
    handle.seek(0)
    previous = handle.read()
    row = {"id": clip_id, "audio": f"audio/{clip_id}.wav", "reference": "", "language": language,
           "tags": [], "required_terms": []}
    handle.seek(0, os.SEEK_END)
    handle.write(("\n" if previous and not previous.endswith("\n") else "") + json.dumps(row) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def record_audio(seconds, ready):
    """Use Scrivo's mic selection/capture and canonical resampler, with no app."""
    from audio_codec import encode_canonical
    from capture import CaptureService, join_capture_blocks

    capture = CaptureService()
    try:
        capture.begin()
        deadline = time.monotonic() + 15
        while capture.is_waking() and time.monotonic() < deadline:
            time.sleep(.02)
        if capture.is_waking() or not capture.is_live():
            raise bench.BenchError("The microphone did not start. No clip was saved.")
        # Discard startup buffers so the timed segment begins at the cue.
        capture.end()
        capture.begin()
        ready()
        time.sleep(seconds)
        samples = capture.end()
        if len(samples) == 0 or not samples.any():
            raise bench.BenchError("No usable microphone audio was captured.")
        canonical = join_capture_blocks([samples], [capture.native_rate], 16000)[:seconds * 16000]
        return encode_canonical(canonical)[0]
    finally:
        capture.shutdown()
        capture.wait_released(5)


def main(argv=None):
    parser = bench.PrivateParser(description="Record one private 16 kHz mono clip; write the reference yourself.")
    parser.add_argument("--id", required=True, metavar="CLIP_ID")
    parser.add_argument("--dir", required=True, metavar="PRIVATE_DIR")
    parser.add_argument("--seconds", type=int, default=15, metavar="SECONDS")
    parser.add_argument("--language", default="en", metavar="LANGUAGE")
    args = parser.parse_args(argv)
    try:
        validate_id(args.id)
        if not 1 <= args.seconds <= 30 or not re.fullmatch(r"[a-z]{2,3}", args.language):
            raise bench.BenchError("Use 1 to 30 seconds and a two- or three-letter language code.")
        folder = bench.outside_repo(Path(args.dir))
        ensure_host()
        with locked_manifest(folder) as handle:
            if any(row["id"] == args.id for row in read_entries(handle)):
                raise bench.BenchError("That clip ID already exists; choose a new ID.")
            audio = folder / "audio" / f"{args.id}.wav"
            # Refuse symlinked audio folders and never replace an earlier take.
            if audio.parent.is_symlink() or audio.exists() or audio.is_symlink():
                raise bench.BenchError("The audio destination is already occupied or symlinked.")
            input("Press Return when ready. Wait for 'Speak now'; Ctrl-C cancels. ")
            cue_fd = os.dup(1)
            try:
                with bench.quiet_backends():
                    pcm = record_audio(args.seconds, lambda: os.write(cue_fd, b"Speak now.\n"))
            finally:
                os.close(cue_fd)
            audio.parent.mkdir(mode=0o700, exist_ok=True)
            fd = os.open(audio, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as raw, wave.open(raw, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(pcm)
            append_entry(handle, args.id, args.language)
        print("Saved one clip and a blank reference slot. Fill the exact reference before benchmarking.")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if isinstance(exc, bench.BenchError):
            print(str(exc), file=sys.stderr)
        else:
            print("Recording did not complete. Check the private folder before retrying; existing files were kept.",
                  file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
