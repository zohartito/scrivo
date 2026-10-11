#!/usr/bin/env python3
"""sotto — hold a key, speak, release; the words land at your cursor. All local.

Gestures (on whichever trigger key you pick; right Option by default):
    hold        push-to-talk — release to transcribe
    double-tap  hands-free — records until the next single tap
    lone tap    ignored (no accidental blips)

    python sotto.py run                      # menu bar app with the pill
    python sotto.py run --trigger fn         # another trigger key
    python sotto.py doctor                   # permission + device checks

The installed Sotto.app asks for the Microphone and Accessibility itself; a
terminal run needs both granted to the terminal. Speech engines (Whisper via
mlx-whisper, Parakeet, Nemotron) run on this Mac; model files download once,
at pinned revisions, and audio never leaves the Mac.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np

import pipeline
# The output guards live in pipeline.py (shared with Windows); these names stay
# importable from sotto for existing callers.
from pipeline import (LOOP_MAX_UNIT_WORDS, LOOP_MIN_PREFIX_CHARS, LOOP_MIN_REPEATS,  # noqa: F401
                      NO_SPEECH_TEXT, _find_repetition_loop, apply_voice_cleanup,
                      finalize_primary_live_delivery, looks_hallucinated, reads_as_no_speech,
                      salvage_repetition_loop, screen)
# The decode options live in speech_config.py (shared with Windows).
from speech_config import transcription_kwargs as _transcription_kwargs
# The log and its rotation live in sotto_log.py (it imports nothing from sotto);
# these names stay importable from sotto for existing callers.
from sotto_log import keep_app_log_small, log, route_app_logs  # noqa: F401
# Delivery to the frontmost app lives in delivery.py (it imports nothing from
# sotto); these names stay importable from sotto for existing callers.
from delivery import (DELIVERY_POLL_S, DELIVERY_WAIT_MAX_S, NOT_IN_HISTORY_EITHER,  # noqa: F401
                      RESTORE_DELAY_S, SOTTO_EVENT_TAG, TRANSIENT_PASTEBOARD_TYPE, DeliveryQueue,
                      PendingDeliveries, character_before_caret, compose_insertion, flush_clipboard_restore,
                      frontmost_pid, inject, is_own_event, own_window_without_text_field, post_command_key,
                      post_own_event, secure_input_active, set_clipboard_text, type_text)
# Shutdown, restart and the drains live in lifecycle.py (it imports nothing
# from sotto); these names stay importable from sotto for existing callers.
from lifecycle import (APP_DRAIN_TIMEOUT, APP_RELAUNCH_SCRIPT, DRAIN_POLL_S, LAUNCHD_EXIT_TIMEOUT_S,  # noqa: F401
                       LAUNCHD_LABEL, QUIT_DRAIN_S, RESTART_DRAIN_DEADLINE_S, RESTART_FAILED_EXIT,
                       RESTART_RELEASE_WAIT_S, SIGTERM_DRAIN_S, SIGTERM_KEEP_RESERVE_S, SIGTERM_RESTORE_WAIT_S,
                       CaptureGate, Lifecycle, RestartController, ShutdownBoundary, app_mode, commit_if_running,
                       dictation_in_flight, end_capture_now, launchd_owns_this_process, launched_as_sealed_job,
                       UPDATE_DRAIN_DEADLINE_S, main_thread_dispatch, relaunch_app_after_exit,
                       shutdown_signal_handlers, wait_until_idle)
# The trigger-key gesture engine lives in gestures.py (it imports nothing from
# sotto); these names stay importable from sotto for existing callers.
from gestures import DOUBLE_TAP_WINDOW_S, HANDS_FREE_MAX_S, HOLD_THRESHOLD_S, GestureEngine  # noqa: F401
# Microphone capture and its CoreAudio helpers live in capture.py (it imports
# nothing from sotto); these names stay importable from sotto for existing callers.
from capture import (HEADPHONE_OUTPUT_TERMINALS, HEADSET_INPUT_TERMINALS, HEADSET_NAME_HINTS,  # noqa: F401
                     IDLE_STOP_S, PREROLL_S, RING_S, SAMPLE_RATE, SILENCE_RESTART_AFTER_S, CaptureService,
                     doctor_input_probe, join_capture_blocks)
# The trigger/hotkey tables and the event-tap listener live in hotkey.py (it
# imports nothing from sotto; run() hands it Quartz as an adapter); these
# names stay importable from sotto for existing callers.
from hotkey import (ALL_MODIFIERS, KEYCODES, MODIFIER_FLAGS, TRIGGER_LABELS, TRIGGERS,  # noqa: F401
                    HotkeyListener, hands_free_hint, hotkey_release_held_seconds, parse_hotkey,
                    trigger_down_in)
# The menu-bar actions live in app_controller.py (it imports nothing from sotto).
from app_controller import AppController
try:  # Keep content-free CLI/admin helpers importable on non-macOS test hosts.
    import Quartz
    from AppKit import NSPasteboard, NSPasteboardTypeString
except ImportError:  # pragma: no cover - capture itself remains macOS-only
    class _UnavailableMacAPI:
        # Constants are declared at module load; capture methods fail only if
        # somebody actually starts the macOS-only runtime.
        def __getattr__(self, name): return 0
    Quartz = _UnavailableMacAPI()
    NSPasteboard = NSPasteboardTypeString = _UnavailableMacAPI()

LONG_CAPTURE_S = 30.0           # beyond one whisper window: collapse dead air
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
HISTORY_KEEP = 200   # every stored transcript is listed; the menu scrolls
FAST_MARGIN_FRAMES = 500         # Fast: encode the speech plus 5 s, not a padded 30 s
MODEL_REWARM_AFTER_S = 240.0

# Sotto keeps its small, required Whisper model on the Mac itself.  The general
# Hugging Face cache may live on removable storage, but dictation must still
# start when that drive is absent.  Set SOTTO_HF_HOME to override this location.
from sotto_paths import MODEL_CACHE_DIR

SOTTO_HF_HOME = MODEL_CACHE_DIR
os.environ["HF_HOME"] = str(SOTTO_HF_HOME)
os.environ["HF_HUB_CACHE"] = str(SOTTO_HF_HOME / "hub")
# No telemetry (README): the Hub client otherwise adds the torch version and
# the calling agent to the user agent of every model download.
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def glossary_source(explicit: str | None) -> Path | None:
    """--glossary wins; otherwise glossary.txt in the data folder, so Sotto.app
    (which takes no arguments) can be taught names and terms too."""
    if explicit:
        return Path(explicit)
    from sotto_paths import DATA_DIR
    candidate = DATA_DIR / "glossary.txt"
    return candidate if candidate.is_file() else None


def sotto_icon_path() -> Path | None:
    """Sotto.app's icon, found from the launcher that started this process."""
    executable = os.environ.get("SOTTO_APP_EXECUTABLE", "").strip()
    icon = Path(executable).parent.parent / "Resources" / "Sotto.icns" if executable else None
    return icon if icon is not None and icon.is_file() else None


ACCESSIBILITY_PROBE = (
    "import ctypes, sys; ax = ctypes.CDLL('/System/Library/Frameworks/ApplicationServices.framework/"
    "ApplicationServices'); ax.AXIsProcessTrusted.restype = ctypes.c_bool; "
    "sys.exit(0 if ax.AXIsProcessTrusted() else 1)")


def accessibility_granted_now() -> bool:
    """AXIsProcessTrusted() asks tccd once per process and then repeats that
    answer, so polling it never sees the switch turn on. A fresh child asks
    again, and macOS attributes it to the same app (Sotto)."""
    import subprocess
    try:
        return subprocess.run([sys.executable, "-I", "-S", "-c", ACCESSIBILITY_PROBE],
                              timeout=5).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def explain_permission(title: str, message: str, pane: str, *, granted=None, register=None) -> bool:
    """In Sotto.app there is no terminal to read: say what is missing and
    offer the exact System Settings pane. With ``granted`` the dialog waits
    and returns True once the permission is on; Quit returns False."""
    import ui
    return ui.permission_dialog(
        title, message, f"x-apple.systempreferences:com.apple.preference.security?{pane}",
        granted=granted, register=register, icon_path=sotto_icon_path())


def persist_engine_and_restart(mode: str, restart, on_failure=None) -> None:
    """A failed restart must not leave the next launch on an untested choice."""
    from speech_config import load_engine_mode, save_engine_mode
    previous = load_engine_mode()
    save_engine_mode(mode)

    def restore(message: str) -> None:
        save_engine_mode(previous)
        if on_failure is not None:
            on_failure(message)

    try:
        if not restart(on_failure=restore):
            raise RuntimeError("Sotto is already shutting down")
    except Exception:
        save_engine_mode(previous)
        raise


def microphone_permission(*, request: bool = False) -> bool:
    """Request audio permission explicitly, before model download/capture."""
    from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
    status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio)
    if status == 3:  # AVAuthorizationStatusAuthorized
        return True
    if status == 0 and request:  # NotDetermined
        completed = threading.Event()
        answer = []

        def received(allowed):
            answer.append(bool(allowed))
            completed.set()

        AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVMediaTypeAudio, received)
        completed.wait(30)
        return bool(answer and answer[0])
    return False


def totals_path():
    import progress
    from sotto_paths import DATA_DIR
    return DATA_DIR / progress.TOTALS_NAME


def record_totals(text: str, seconds: float) -> None:
    """Lifetime counts for Your progress: words and seconds, never text."""
    import progress
    try:
        progress.record(totals_path(), text=text, seconds=seconds)
    except (OSError, ValueError) as exc:  # unreadable or malformed: keep the file, skip this one
        log(f"! progress totals not saved ({type(exc).__name__})")


def seed_totals(store) -> None:
    import progress
    try:
        progress.ensure_totals(totals_path(), store.entries(limit=HISTORY_KEEP))
    except OSError as exc:
        log(f"! progress totals not started ({type(exc).__name__})")


def apply_personal_dictionary(text: str, preprocessing: dict) -> str:
    """pipeline.apply_personal_dictionary, logging to Sotto's log."""
    return pipeline.apply_personal_dictionary(text, preprocessing, log)


def live_preview(text: str) -> str:
    """The pill's live words with the user's spellings, so they match what gets pasted."""
    if not text:
        return text
    import dictionary
    try:
        return dictionary.apply(text, dictionary.load(dictionary.DICTIONARY_PATH))[0]
    except Exception:  # display only: a broken dictionary leaves the words as heard
        return text


INSTALL_TIMEOUT_S = 3600.0  # an engine download on a slow link, but never forever


def run_installer(argv: list[str], *, run=subprocess.run) -> tuple[bool, str]:
    """Run an optional-engine installer; (succeeded, its last output). Never
    raises, so the caller always clears its "installing" state."""
    try:
        result = run(argv, capture_output=True, text=True, timeout=INSTALL_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {INSTALL_TIMEOUT_S / 60:.0f} minutes"
    except OSError as exc:
        return False, f"could not start ({exc.strerror or exc})"
    return result.returncode == 0, (result.stderr or result.stdout).strip()[-200:]


def desktop_audio_path(folder: Path, timestamp: float) -> Path:
    """Where "Save audio to Desktop" writes a recording: named by the second
    it was captured, with -2, -3, … so an earlier save is never overwritten.

    Side effects: creates the file empty with exclusive create, so two saves
    running at once can never pick the same name.
    """
    import datetime
    stem = f"sotto-{datetime.datetime.fromtimestamp(timestamp):%Y%m%d-%H%M%S}"
    candidate, number = folder / f"{stem}.wav", 2
    while True:
        try:
            with open(candidate, "xb"):
                return candidate
        except FileExistsError:
            candidate, number = folder / f"{stem}-{number}.wav", number + 1


def save_audio_copy(source: Path, folder: Path, timestamp: float) -> Path:
    """Copy a recording into folder under a fresh desktop_audio_path name.

    Side effects: writes one new file. The source is opened first, so a
    missing recording raises FileNotFoundError without leaving a file behind.
    """
    with open(source, "rb") as audio:
        target = desktop_audio_path(folder, timestamp)
        with open(target, "wb") as saved:
            shutil.copyfileobj(audio, saved)
    return target


def choose_language(probabilities: dict, allowed) -> str:
    """Automatic's pick. Prefer the user's languages, but speech that is
    clearly another language (the allowed ones score < 0.25 while it scores
    >= 0.5) is written in that language: forcing it into an allowed one makes
    Whisper translate it. Measured: en/he speech scores >= 0.989 in-set;
    other languages <= 0.112 in-set with >= 0.863 for themselves."""
    top = max(probabilities, key=probabilities.get)
    if not allowed:
        return top
    best = max(allowed, key=lambda code: probabilities.get(code, 0.0))
    if top not in allowed and probabilities.get(best, 0.0) < 0.25 and probabilities[top] >= 0.5:
        return top
    return best


class ModelUnavailable(RuntimeError):
    """A plain, actionable reason the Whisper model cannot be used yet."""


def _complete_whisper_snapshot(path) -> str:
    """An interrupted download can leave a snapshot folder without weights."""
    folder = Path(path)
    if (folder / "config.json").is_file() and any(
            (folder / name).is_file() for name in ("weights.safetensors", "weights.npz")):
        return str(folder)
    raise FileNotFoundError(f"incomplete model snapshot: {folder}")


def resolve_pinned_snapshot(repo: str, allow_patterns: list, complete) -> str:
    """The local snapshot of ``repo`` at its pinned commit, cache-first.

    A cached snapshot is used without any Hub request; a first run that is
    not offline downloads exactly the pinned bytes; offline flags forbid it.
    ``complete`` rejects a folder an interrupted download left partial."""
    from offline_runtime import offline_requested
    from huggingface_hub import snapshot_download
    from speech_config import MODEL_REVISIONS
    options = {"repo_id": repo, "revision": MODEL_REVISIONS.get(repo),
               "cache_dir": str(SOTTO_HF_HOME / "hub"), "allow_patterns": allow_patterns}
    try:
        return complete(snapshot_download(**options, local_files_only=True))
    except FileNotFoundError:
        if offline_requested():
            raise ModelUnavailable(
                f"{repo} is not downloaded yet. Run once without SOTTO_OFFLINE, "
                "HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE to download it.") from None
    try:
        return complete(snapshot_download(**options))
    except Exception as exc:
        raise ModelUnavailable(
            f"could not download {repo} ({type(exc).__name__}: {str(exc)[:160]}). "
            "Check internet access and free disk space, then run again.") from None


def parakeet_cached() -> bool:
    """Is the pinned Parakeet snapshot fully downloaded? (a path check only)"""
    from speech_config import MODEL_PROFILES, pinned_snapshot_cached
    return pinned_snapshot_cached(MODEL_PROFILES["parakeet"].repo, SOTTO_HF_HOME / "hub",
                                  ("config.json", "model.safetensors"))


def _complete_parakeet_snapshot(path) -> str:
    folder = Path(path)
    if (folder / "config.json").is_file() and (folder / "model.safetensors").is_file():
        return str(folder)
    raise FileNotFoundError(f"incomplete model snapshot: {folder}")


class LocalParakeet:
    """Parakeet TDT 0.6B v3: the fastest engine, 25 European languages detected
    by the model itself. Loaded from the pinned local snapshot (never through
    a Hub id) and fed the audio array directly, so no ffmpeg is involved."""

    CHUNK_SECONDS = 120  # long hands-free captures are decoded in pieces

    def __init__(self, repo: str):
        self.path = resolve_pinned_snapshot(repo, ["config.json", "model.safetensors"],
                                            _complete_parakeet_snapshot)
        self._model = None
        self.last_language = None  # the model does not report one

    def _load(self):
        if self._model is None:
            import json
            import mlx.core as mx
            from mlx.utils import tree_flatten, tree_unflatten
            from parakeet_mlx.utils import from_config
            model = from_config(json.loads((Path(self.path) / "config.json").read_text()))
            model.load_weights(str(Path(self.path) / "model.safetensors"))
            model.update(tree_unflatten([(name, value.astype(mx.bfloat16))
                                         for name, value in tree_flatten(model.parameters())]))
            self._model = model
        return self._model

    def transcribe(self, samples, **_whisper_options):
        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel
        model = self._load()
        audio = np.asarray(samples, dtype=np.float32)
        step = self.CHUNK_SECONDS * SAMPLE_RATE
        pieces = []
        for start in range(0, max(len(audio), 1), step):
            chunk = audio[start:start + step]
            if len(chunk) < SAMPLE_RATE // 10:
                continue
            mel = get_logmel(mx.array(chunk), model.preprocessor_config)
            pieces.append(model.generate(mel)[0].text.strip())
        return {"text": " ".join(piece for piece in pieces if piece), "segments": [], "language": None}


class LocalWhisper:
    """Resolve/download once during startup; inference only sees a local path.

    A cached snapshot of the pinned revision is used without any Hub request,
    so an already-downloaded model never touches the network."""

    def __init__(self, backend, repo: str):
        candidate = Path(repo).expanduser()
        if candidate.is_dir():
            self.path = str(candidate.absolute())
        else:
            self.path = resolve_pinned_snapshot(repo, ["*.json", "*.safetensors", "*.npz"],
                                                _complete_whisper_snapshot)
        self.backend = backend

    allowed_languages = None  # callable -> the codes Automatic may choose among
    speed = None              # callable -> "accurate" | "fast" (Settings)
    last_language = None

    def _features(self, model, samples, *, trim: bool):
        """Encoder features for one window. ``trim`` (Fast) encodes only the
        speech plus FAST_MARGIN_FRAMES instead of padding to 30 s — about 3x
        faster; experimental, so every result still passes the unsure check."""
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
        mel = log_mel_spectrogram(np.asarray(samples, dtype=np.float32),
                                  n_mels=model.dims.n_mels, padding=N_SAMPLES)
        content = max(mel.shape[-2] - N_FRAMES, 1)
        if not trim:
            return model.encoder(pad_or_trim(mel[:content], N_FRAMES, axis=-2).astype(mx.float16)[None])
        frames = min(N_FRAMES, content + FAST_MARGIN_FRAMES)
        frames += frames % 2
        encoder = model.encoder
        x = nn.gelu(encoder.conv1(mel[:frames].astype(mx.float16)[None]))
        x = nn.gelu(encoder.conv2(x))
        x = x + encoder._positional_embedding[: x.shape[1]]
        for block in encoder.blocks:
            x, _, _ = block(x)
        return encoder.ln_post(x)

    @staticmethod
    def _language_probabilities(model, features) -> dict:
        """Whisper's language-detection math for any encoder context length."""
        import mlx.core as mx
        from mlx_whisper.tokenizer import get_tokenizer
        tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages)
        logits = model.logits(mx.array([[tokenizer.sot]]), features)[:, 0]
        mask = mx.full(logits.shape[-1], -mx.inf, dtype=mx.float32)
        mask[list(tokenizer.all_language_tokens)] = 0.0
        probabilities = np.array(mx.softmax(logits + mask, axis=-1))[0]
        return {code: float(probabilities[token]) for token, code in
                zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)}

    def _model(self):
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder
        return ModelHolder.get_model(self.path, mx.float16)  # the instance transcribe() uses

    def detect_language(self, samples, allowed) -> str:
        """Most likely language among ``allowed``, from the first 30 s."""
        model = self._model()
        probabilities = self._language_probabilities(model, self._features(model, samples, trim=False))
        return choose_language(probabilities, allowed)

    def one_pass(self, samples, allowed, *, language=None, trim=False):
        """A clip of at most 30 s: ONE encoder pass serves the language choice
        (when ``language`` is None) and a greedy decode. Returns (language,
        result or None); None hands the clip to the full pipeline, whose
        temperature fallback handles hard audio, using Whisper's thresholds."""
        from mlx_whisper.decoding import DecodingOptions, DecodingTask
        model = self._model()
        features = self._features(model, samples, trim=trim)
        if language is None:
            language = choose_language(self._language_probabilities(model, features), allowed)
        task = DecodingTask(model, DecodingOptions(language=language, without_timestamps=True, temperature=0.0))
        task._get_audio_features = lambda audio_features: audio_features  # already encoded
        result = task.run(features)[0]
        if len(result.tokens) >= task.sample_len - 1:
            return language, None  # budget exhausted: never paste a cut-off prefix
        if result.no_speech_prob > 0.6 and result.avg_logprob <= -1.0:
            return language, {"text": "", "language": language, "segments": []}  # Whisper's skip rule
        if result.compression_ratio > 2.4 or result.avg_logprob < -1.0:
            return language, None
        return language, {"text": result.text, "language": language, "segments": []}

    def transcribe(self, samples, **kwargs):
        kwargs["path_or_hf_repo"] = self.path
        self.last_language = None
        explicit = kwargs.get("language")
        allowed = () if explicit else tuple(self.allowed_languages() if self.allowed_languages else ())
        language = explicit or (allowed[0] if len(allowed) == 1 else None)
        fast = bool(self.speed and self.speed() == "fast")
        short = len(samples) <= 30 * SAMPLE_RATE and "initial_prompt" not in kwargs
        if short and (fast or (language is None and allowed)):
            chosen, result = self.one_pass(samples, allowed, language=language, trim=fast)
            if not explicit:
                self.last_language = chosen
            if result is not None:
                return result
            kwargs["language"] = chosen
        elif language is not None:
            kwargs["language"] = language
            if not explicit:
                self.last_language = language
        elif allowed:
            kwargs["language"] = self.last_language = self.detect_language(samples, allowed)
        return self.backend.transcribe(samples, **kwargs)


def model_rewarm_due(last_finished: float, now: float, *, rewarming: bool,
                     queue_idle: bool, adaptive: bool) -> bool:
    """Whether a baseline Metal refresh should hide inside the next capture."""
    return (not adaptive and not rewarming and queue_idle
            and now - last_finished >= MODEL_REWARM_AFTER_S)


def prepare_for_whisper(raw: np.ndarray, native_rate: float) -> np.ndarray:
    """Native capture -> 16 kHz normalized whisper input: polyphase resample
    (linear interp aliased audibly and cost accuracy), then robust
    percentile normalization in both directions with glitch-spike clipping."""
    if not len(raw):
        return raw
    if native_rate <= 0:
        log("! capture had no valid sample rate — dropped")
        return np.zeros(0, dtype=np.float32)
    if native_rate != SAMPLE_RATE:
        from fractions import Fraction

        from scipy.signal import resample_poly
        ratio = Fraction(SAMPLE_RATE, int(native_rate)).limit_denominator(1000)
        raw = resample_poly(raw, ratio.numerator,
                            ratio.denominator).astype(np.float32)
    scale = float(np.percentile(np.abs(raw), 99.9))
    peak = float(np.abs(raw).max())
    if peak > 2 * max(scale, 1e-6):
        log(f"! input glitch spike clipped (peak {peak:.1f}, "
            f"speech level {scale:.3f})")
    if scale > 1e-4:
        raw = raw * (0.5 / scale)
    return np.clip(raw, -1.0, 1.0)


DEAD_ROUTE_TEXT = "[dead microphone]"


def is_dead_route(samples: np.ndarray) -> bool:
    """Bit-exact silence is a dead capture path, not a quiet room.

    CaptureService._tap already treats exact-zero buffers as a dead route:
    a live mic always carries a noise floor. Quiet whispered dictation is
    non-zero (2026-08-13/14) and must still reach the ASR. Exact zeros must
    not: Whisper turns them into repetition loops (2026-09-20: 40.96s of
    24 kHz zeros -> "ощ" x223).
    """
    if samples is None or len(samples) == 0:
        return True
    return float(np.max(np.abs(samples))) == 0.0


def asr_skip_reason(samples: np.ndarray) -> str | None:
    """Why this capture must not be sent to the ASR, or None to transcribe."""
    if is_dead_route(samples):
        return "dead microphone (all-zero capture)"
    return None


def collapse_silence(samples: np.ndarray) -> np.ndarray:
    """Shorten long silent stretches in a capture. Dead air is what whisper
    hallucinates on during long dictations ("Okay. Okay. Okay."), and it
    slows inference. The threshold adapts to the recording's own noise floor,
    and up to 1s of every pause is kept so sentence rhythm survives."""
    block = int(SAMPLE_RATE * 0.1)
    n_blocks = len(samples) // block
    if n_blocks < 20:
        return samples
    body = samples[:n_blocks * block].reshape(n_blocks, block)
    rms = np.sqrt(np.mean(body**2, axis=1))
    # adapt to the noise floor, but never above half the loud level — on
    # uniform wall-to-wall speech nothing must qualify as silence
    threshold = max(0.002, min(float(np.percentile(rms, 10)) * 2.0,
                               float(np.percentile(rms, 90)) * 0.5))
    keep = np.ones(n_blocks, dtype=bool)
    run = 0
    for i, level in enumerate(rms):
        run = run + 1 if level < threshold else 0
        if run > 10:  # keep at most 1.0s of any silent stretch
            keep[i] = False
    if keep.all():
        return samples
    return np.concatenate([body[keep].reshape(-1), samples[n_blocks * block:]])


# -- daemon ------------------------------------------------------------------

def overlay_learning_state(entries: list[dict], active_history_ids: set[str]) -> list[dict]:
    """Return history rows with the UI-only learning-state overlay applied."""
    return [dict(entry, learning_state="active") if entry.get("id") in active_history_ids
            else dict(entry) for entry in entries]


def _active_learning_history_ids(coordinator) -> set[str]:
    """Read the coordinator-owned active corpus without exposing its text."""
    with coordinator._lock:  # coordinator is the serialization boundary
        return {record["history_id"] for record in coordinator.learning.active()
                if isinstance(record.get("history_id"), str)}


def learning_status_text(snapshots) -> str:
    """Small headless status surface; deliberately contains no transcript text."""
    return f"{len(snapshots)} active learning item(s)"


def learning_list_text(snapshots) -> str:
    """Format active learning identities only; references remain private."""
    if not snapshots:
        return "0 active learning item(s)"
    return "\n".join(
        f"sample_id={snapshot.sample_id} history_id={snapshot.history_id}"
        for snapshot in sorted(snapshots, key=lambda snapshot: snapshot.sample_id)
    )


def remove_learning_sample(sample_id: str, *, history_id_lookup, adaptive_revoke, artifact_revoke) -> bool:
    """Content-free ordering seam for the offline admin command."""
    history_id = history_id_lookup(sample_id)
    if history_id:
        adaptive_revoke(history_id)
    return bool(artifact_revoke(sample_id))


def nonadaptive_correct(dependency_guard, coordinator, entry_id: str, text: str, *, expected_revision: int | None = None):
    """Invalidate adaptive evidence before an ordinary correction mutates it."""
    return coordinator.review_transaction(
        entry_id, expected_revision,
        lambda: dependency_guard.revoke(entry_id, reason="reference_changed_nonadaptive"),
        lambda: (coordinator.correct(entry_id, text, expected_revision=expected_revision)
                 if expected_revision is not None else coordinator.correct(entry_id, text)))


def nonadaptive_delete(dependency_guard, coordinator, entry_id: str):
    dependency_guard.revoke(entry_id, reason="history_deleted")
    return coordinator.delete(entry_id)


def nonadaptive_clear(dependency_guard, coordinator) -> None:
    dependency_guard.revoke_all(reason="history_cleared")
    coordinator.clear()


def nonadaptive_revoke_learning(dependency_guard, coordinator, entry_id: str):
    dependency_guard.revoke(entry_id, reason="learning_revoked")
    return coordinator.revoke_history(entry_id)


HISTORY_UNREADABLE_TITLE = "History could not be read"
LEARNING_UNREADABLE_TITLE = "Learning set could not be read"


class UnsavedHistory:
    """Stands in for LearningCoordinator while history.jsonl (F16b), or the
    learning set's learning.jsonl or pending-gold.jsonl (N21), cannot be read.

    Dictation keeps working: ``append_live`` hands back an in-memory row so the
    text is still delivered, but nothing (row, audio, learning state) is
    written. Every other coordinator operation raises the store's
    HistoryUnreadable, exactly as the real coordinator would on its re-read,
    so Retry, Correct, Delete and Clear can never act on a partial view."""

    def __init__(self, store, learning) -> None:
        self.history, self.learning = store, learning
        self._lock = threading.RLock()

    def append_live(self, text: str, samples, duration: float, model: str, **metadata) -> dict:
        """Side effects: none; the row and its audio exist only in memory."""
        return {"id": metadata.get("entry_id") or uuid.uuid4().hex[:12], "ts": metadata.get("ts"),
                "text": text, "duration": round(duration, 2), "model": model, "saved": False}

    def __getattr__(self, name: str):
        from history import HistoryUnreadable
        if self.history.unreadable is not None:
            raise HistoryUnreadable(f"History is unreadable and was left unchanged ({self.history.unreadable})")
        raise HistoryUnreadable("History is not changed while the learning set is unreadable "
                                f"({self.learning.unreadable})")


def history_coordinator(store, learning_store):
    """The dictation session's coordinator for ``store``.

    A store opened with ``tolerate_unreadable=True`` over an unreadable
    history.jsonl (F16b), or a learning store opened the same way over an
    unreadable learning set (N21), gets an UnsavedHistory instead of a
    LearningCoordinator, whose startup re-read would raise before the menu
    bar exists; the caller tells the user once with
    ``unreadable_store_alert``. Side effects: a LearningCoordinator
    validates the active learning links against History."""
    if store.unreadable is not None or learning_store.unreadable is not None:
        return UnsavedHistory(store, learning_store)
    from learning import LearningCoordinator
    return LearningCoordinator(store, learning_store)


def history_unreadable_message(store) -> str:
    # Never suggest moving or deleting the file: a launch with no index would
    # treat every kept recording as an orphan and sweep it.
    cause, remedy = (("it was written by a newer version of Sotto", "update Sotto again")
                     if store.unreadable.newer_schema else
                     ("part of it is damaged", "repair the damaged line"))
    return (f"Sotto cannot read {store.index} because {cause}. The file and its recordings "
            "were left exactly as they are. Dictation still works, but new dictations are "
            f"not saved to History until the file can be read again: {remedy}, then restart Sotto.")


def learning_unreadable_message(learning_store) -> str:
    # As for History: never suggest moving or deleting the file, which would
    # make every kept learning recording look like an orphan.
    return (f"Sotto cannot read {learning_store.unreadable.path} because part of it is damaged. "
            "Your learning set and History were left exactly as they are. Dictation still works, but "
            "new dictations are not saved to History until the file can be read again: repair the "
            "damaged line, then restart Sotto.")


def unreadable_store_alert(store, learning_store) -> tuple[str, str] | None:
    """(title, message) for the one startup alert about an unreadable History
    or learning set, or None when both read."""
    if store.unreadable is not None:
        return HISTORY_UNREADABLE_TITLE, history_unreadable_message(store)
    if learning_store.unreadable is not None:
        return LEARNING_UNREADABLE_TITLE, learning_unreadable_message(learning_store)
    return None


# A live capture Sotto stopped before transcribing (N9, N9b): its audio is
# kept in History under this text so Retry can transcribe it.
NOT_TRANSCRIBED_TEXT = "[not transcribed: Sotto stopped]"


def append_untranscribed_capture(coordinator, *, raw, native_rate, captured_ts, model, speech_config,
                                 glossary_terms, queue_wait_s, error: str) -> dict:
    """Append the ``live_suspect`` row NOT_TRANSCRIBED_TEXT for a capture that
    was never transcribed, with its canonical and raw audio, for Retry. The
    N9 teardown keep and the N9b spool recovery share it.

    Side effects: one History append (raises what the append raises)."""
    from audio_codec import prepare_canonical
    prepared = prepare_canonical(prepare_for_whisper(raw, native_rate))
    seconds = len(prepared.asr_samples) / SAMPLE_RATE
    _, attempt_metadata = _transcription_kwargs(speech_config, glossary_terms, seconds)
    attempt_metadata.update({
        "vad": {"available": None, "speech_fraction": None, "span_count": None},
        "preprocessing": {"resampled_normalized": True, "vad_trimmed": False,
                          "silence_collapsed": False, "outcome": "suspect", "error": error},
        "latency": {"queue_wait_seconds": queue_wait_s, "asr_seconds": 0.0},
    })
    return coordinator.append_live(
        NOT_TRANSCRIBED_TEXT, prepared, seconds, model, ts=captured_ts,
        raw_samples=raw, raw_sample_rate=native_rate, provenance="live_suspect",
        adaptive=False, **attempt_metadata)


def recover_capture_spool(spool, store, coordinator, *, model, speech_config, glossary_terms, log) -> int:
    """At launch, before the UI: every capture a stopped Sotto left in the
    spool (N9b: a crash, SIGKILL or a logout that skipped the drain) becomes
    a NOT_TRANSCRIBED_TEXT History row with its audio, exactly as the N9
    teardown keep would have saved it; its file goes only once that row is
    saved. A capture whose row was saved just before the stop (same key-up
    time) is not added twice. While History cannot be written (read-only,
    F16b/N21) nothing is touched. Returns how many captures still wait.

    Side effects: History appends; unlinks recovered and partial spool files; logs."""
    waiting = spool.leftovers()
    if isinstance(coordinator, UnsavedHistory):
        if waiting:
            log(f"! {len(waiting)} recording(s) from when Sotto last stopped wait in {spool.directory} "
                "until History can be read again")
        return len(waiting)
    spool.discard_partial()
    saved_ms = {round(row["ts"] * 1000) for row in store.entries(limit=sys.maxsize)
                if isinstance(row.get("ts"), (int, float))}
    recovered, left = 0, 0
    for item in waiting:
        if item.captured_ms in saved_ms:
            spool.remove(item)
            continue
        try:
            raw, rate = spool.load(item)
            append_untranscribed_capture(
                coordinator, raw=raw, native_rate=float(rate), captured_ts=item.captured_ts,
                model=model, speech_config=speech_config, glossary_terms=glossary_terms, queue_wait_s=None,
                error="Sotto stopped before transcribing; recovered from the capture spool at the next launch")
        except Exception as exc:
            left += 1
            log(f"! a recording from when Sotto last stopped could not be added to History "
                f"({type(exc).__name__}: {str(exc)[:120]}); it stays in {spool.directory}")
            continue
        spool.remove(item)
        recovered += 1
    if recovered:
        log(f"○ {recovered} recording(s) Sotto stopped before transcribing are in History for Retry")
    return left


def spool_waiting_note(waiting: int) -> str:
    """The sentence the unreadable-History alert adds while recordings wait (N9b)."""
    return (f"\n\n{waiting} recording(s) from when Sotto last stopped are kept on disk and will be "
            "added to History once it can be read again." if waiting else "")


def _read_only_json(path: Path) -> dict | None:
    """Read a strict regular JSON record without constructing any store."""
    try:
        mode=os.lstat(path).st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode): return None
        value=json.loads(path.read_text("utf-8"))
        return value if isinstance(value,dict) else None
    except (OSError,ValueError,json.JSONDecodeError):
        return None


def _read_only_last_jsonl(path: Path) -> dict | None:
    """Return the exact final deployment record; malformed tails fail closed."""
    try:
        mode=os.lstat(path).st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode): return None
        lines=[line for line in path.read_text("utf-8").splitlines() if line.strip()]
        value=json.loads(lines[-1]) if lines else None
        return value if isinstance(value,dict) else None
    except (OSError,ValueError,json.JSONDecodeError):
        return None


def adaptive_status_text(snapshots, *, base_dir: Path | None = None) -> str:
    """Read content-free adaptive/silver/deployment status without stores."""
    from history import STORE_DIR
    root=base_dir or STORE_DIR
    state = _read_only_json(root / "adaptive-learning" / "state.json")
    if state is None:
        return learning_status_text(snapshots) + "\nadaptive English: collecting development 0/20"
    captures = state.get("captures", {}) if isinstance(state, dict) else {}
    dev = sum(1 for row in captures.values() if isinstance(row, dict) and row.get("outcome") in {"corrected", "correct_as_is"} and not row.get("pool_generation"))
    current = ((state.get("champions", {}) or {}).get("current") or (state.get("champions", {}) or {}).get("baseline") or {}).get("stable_id")
    active = [g for g in (state.get("generations", {}) or {}).values() if isinstance(g, dict) and g.get("status") in {"pool_open", "pool_closed", "evaluating"}]
    if active:
        g = active[-1]; stage = "evaluating" if g.get("status") in {"pool_closed", "evaluating"} else f"reviewing pool {len(g.get('pool', []))}/33"
    else:
        stage = f"collecting development {dev}/20"
    try:
        from maintainer.silver_store import read_only_status
        silver=read_only_status(root)
        deploy=_read_only_last_jsonl(root / "adaptive-learning" / "silver" / "deployment.jsonl")
        if not isinstance(deploy,dict) or deploy.get("schema") != 1 or not isinstance(deploy.get("tier"),str):
            raise ValueError("deployment metadata malformed")
        horizon={}
        if silver.get("state") != "ok":
            raise RuntimeError("silver status blocked")
        jobs=silver.get("jobs",{}); worker=silver.get("worker") or {}; canary=deploy.get("canary") or {}
        silver_text=(f" · silver={silver['accepted']} accepted/{silver['abstained']} abstained"
                     f" queued={jobs.get('pending',0)} processing={jobs.get('leased',0)} discarded={jobs.get('discarded',0)}"
                     f" · worker={worker.get('state','unknown')}"
                     f" · horizon={horizon.get('status','none')} {horizon.get('members',0)}/500"
                     f" · deployment={deploy['tier']} {canary.get('stage','')}" )
    except Exception:
        silver_text=" · silver=unavailable (fail closed)"
    return learning_status_text(snapshots) + f"\nadaptive English: {stage}" + (f" · champion={current}" if current else "") + silver_text


def export_learning_snapshot(output: str | Path) -> int:
    """Copy consented corpus artifacts and write a benchmark-compatible manifest.

    The storage guard validates active snapshots and prevents cross-process
    revocation from unlinking a source while it is being copied.  We copy source
    WAV bytes verbatim: export is never a decode/re-encode operation.
    """
    from learning import LearningStore

    destination = Path(output).expanduser()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"Export destination already exists and is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        with LearningStore.admin_export_guard() as snapshots:
            # The pre-guard check is a fast failure.  This one is the race-free
            # decision: a concurrent exporter that observed an empty directory
            # must not add artifacts after the first guarded export succeeds.
            if not destination.is_dir() or any(destination.iterdir()):
                raise ValueError(
                    f"Export destination already exists and is not empty: {destination}")
            for snapshot in snapshots:
                audio_relative = Path("audio") / f"{snapshot.sample_id}.wav"
                audio_target = destination / audio_relative
                audio_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(snapshot.inference_audio_path, audio_target)
                row = {
                    "id": snapshot.sample_id,
                    "audio": audio_relative.as_posix(),
                    "reference": snapshot.corrected_text,
                    "tags": [],
                    "required_terms": [],
                    "language": None,
                    "provenance": {
                        "history_id": snapshot.history_id,
                        "history_revision": snapshot.history_revision,
                        "inference_sha256": snapshot.inference_identity.sha256,
                        "inference_format": snapshot.inference_identity.format,
                    },
                }
                if snapshot.raw_audio_path is not None:
                    raw_relative = Path("raw") / f"{snapshot.sample_id}.wav"
                    raw_target = destination / raw_relative
                    raw_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(snapshot.raw_audio_path, raw_target)
                    row["raw"] = raw_relative.as_posix()
                    row["provenance"]["raw_sha256"] = snapshot.raw_identity.sha256
                rows.append(row)
            manifest = destination / "manifest.jsonl"
            with manifest.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        # The destination was required to be empty, so a partial export must
        # remain visible for inspection rather than deleting user-selected data.
        raise
    return len(rows)


def transcribe_canonical_samples(mlx_whisper, samples: np.ndarray, speech_config,
                                 glossary_terms: tuple[str, ...], *, nemotron=None) -> tuple[str, dict]:
    """Transcribe exact canonical samples once, with the resolved local config."""
    duration = len(samples) / SAMPLE_RATE
    if speech_config.profile.backend == "nemotron":
        from nemotron_backend import metadata
        if nemotron is None:
            raise RuntimeError("Nemotron runtime is required")
        return nemotron.transcribe(samples), metadata()
    kwargs, metadata = _transcription_kwargs(speech_config, glossary_terms, duration)
    result = mlx_whisper.transcribe(samples, **kwargs)
    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
        raise RuntimeError("mlx_whisper.transcribe returned no text")
    chosen = getattr(mlx_whisper, "last_language", None)
    if speech_config.language is None and isinstance(chosen, str):
        metadata["detected_language"] = chosen  # Automatic's pick among the allowed set
    return result["text"].strip(), metadata


def transcribe_prepared(mlx_whisper, prepared, speech_config, glossary_terms: tuple[str, ...]) -> tuple[str, dict]:
    """Transcribe a :class:`PreparedAudio` without changing its ASR samples."""
    return transcribe_canonical_samples(mlx_whisper, prepared.asr_samples,
                                        speech_config, glossary_terms)


def stage_adaptive_live_audio(history, prepared, capture_id: str):
    """Bind live routing to the exact private canonical WAV that will persist.

    The normal push-to-talk path must not route a mutable in-memory array and
    then write an unrelated canonical file after inference.  The capture id
    is already the durable request identity, so its owned History path is a
    safe pre-publication location.  ``append_live(..., entry_id=capture_id)``
    later verifies and republishes these same prepared PCM bytes atomically
    with history metadata.  An exception caller removes this unadopted file.
    """
    from audio_codec import read_canonical_wav, write_canonical_wav
    if not isinstance(capture_id,str) or not capture_id:
        raise ValueError("invalid live capture identity")
    path=history.audio_path(capture_id)
    write_canonical_wav(path,prepared.pcm)
    _samples,identity=read_canonical_wav(path)
    if identity != prepared.identity:
        try: path.unlink(missing_ok=True)
        except OSError: pass
        raise RuntimeError("staged live canonical identity mismatch")
    return path


def discard_staged_adaptive_live_audio(history, capture_id: str) -> None:
    """Remove only an unadopted direct owned live staging file."""
    try:
        path=history.audio_path(capture_id)
        if history.get(capture_id) is None and path.exists() and not path.is_symlink():
            path.unlink();
    except (OSError,ValueError):
        pass


def adaptive_live_transcribe_prepared(runtime, prepared, capture_id: str):
    """Primary push-to-talk adaptive call with immutable canonical authority."""
    path=stage_adaptive_live_audio(runtime.history,prepared,capture_id)
    try:
        text,metadata=runtime.live_transcribe(prepared,canonical_path=path,
                                              canonical_identity=prepared.identity,capture_id=capture_id)
        return text,metadata,path
    except Exception:
        discard_staged_adaptive_live_audio(runtime.history,capture_id)
        raise


def warm_speech_runtime(*, adaptive: bool, model: str, speech_config, mlx_whisper=None, adaptive_runtime=None) -> None:
    """Injectable warmup seam with no adaptive routing side effects."""
    silence = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
    if adaptive:
        if adaptive_runtime is None: raise RuntimeError("adaptive runtime is required")
        adaptive_runtime.warmup_baseline(silence)
        return
    if mlx_whisper is None: raise RuntimeError("mlx_whisper is required")
    kwargs = {"path_or_hf_repo": model}
    if speech_config.language is not None: kwargs["language"] = speech_config.language
    mlx_whisper.transcribe(silence, **kwargs)

def run(trigger: str, model: str | None = None, overlay: bool = True,
        hotkey_spec: str = "ctrl-opt-d", idle_release: float = IDLE_STOP_S,
        speech_config=None, glossary_terms: tuple[str, ...] = (), adaptive: bool = False,
        engine_switching: bool = True, settings_locked: frozenset | set = frozenset()) -> None:
    """Run capture using a resolved config, without importing MLX before setup."""
    from speech_config import language_mode, resolve_speech_config, save_language_mode, with_language
    if speech_config is None:
        speech_config = resolve_speech_config(model=model)
    model = speech_config.model_repo
    use_nemotron = speech_config.profile.backend == "nemotron"
    use_parakeet = speech_config.profile.backend == "parakeet"
    if adaptive and (use_nemotron or use_parakeet):
        raise ValueError("Nemotron and Parakeet are manual engines; they cannot enable adaptive routing")
    language_lock = threading.Lock()
    language_state = {"mode": language_mode(speech_config.language)}

    def current_speech_config():
        with language_lock:
            return with_language(speech_config, language_state["mode"])

    shutdown = ShutdownBoundary()
    restart = RestartController(shutdown)
    import ApplicationServices

    # Fail fast (before the slow model load) so a launchd restart loop is cheap,
    # and let macOS open the grant prompt for whatever binary is running us.
    if app_mode():
        import ui
        ui.use_app_icon(sotto_icon_path())
    if app_mode() and not ApplicationServices.AXIsProcessTrusted():
        # Sotto.app: explain, open the right pane, and continue once granted.
        if not explain_permission(
                "Sotto needs Accessibility",
                "Click Open System Settings and turn on Sotto under Privacy & Security → "
                "Accessibility, so it can hear your dictation key and put text at the cursor. "
                "This message closes by itself once it is on.\n\nAlready on? Sotto was rebuilt: "
                "select it, remove it with −, then add it again with +.",
                "Privacy_Accessibility",
                granted=accessibility_granted_now,
                register=lambda: ApplicationServices.AXIsProcessTrustedWithOptions(
                    {ApplicationServices.kAXTrustedCheckOptionPrompt: True})):
            log("accessibility not granted — open Sotto again after turning it on")
            sys.exit(0)  # a clean exit: launchd must not loop this dialog
        # This process still holds the cached "not trusted": start a fresh one.
        log("accessibility granted — restarting")
        relaunch_app_after_exit()
        sys.exit(0)
    elif not ApplicationServices.AXIsProcessTrustedWithOptions(
            {ApplicationServices.kAXTrustedCheckOptionPrompt: True}):
        log("accessibility not granted — grant it in System Settings "
            "(the entry may appear as 'Python'), then run the command again")
        sys.exit(1)

    if not microphone_permission(request=True):
        if app_mode():
            explain_permission(
                "Sotto needs the microphone",
                "Turn on Sotto under Privacy & Security → Microphone, then open Sotto again. "
                "It listens only while you hold the dictation key.",
                "Privacy_Microphone")
            sys.exit(0)
        log("microphone not granted — System Settings → Privacy & Security → "
            "Microphone → enable your terminal/Python, then run the command again")
        sys.exit(1)

    from PyObjCTools import AppHelper
    mlx_whisper = None
    nemotron = None
    if use_nemotron:
        from nemotron_backend import NemotronRuntime
        log("warming up Nemotron English streaming ...")
        try:
            nemotron = NemotronRuntime()
            log("✓ Nemotron model ready")
        except Exception as exc:
            # An optional backend must not strand the working dictation app
            # in a launchd crash loop after a runtime/model failure.
            from speech_config import load_language_mode, save_engine_mode
            log(f"! Nemotron unavailable ({str(exc)[:120]}) — restoring Whisper")
            try:
                save_engine_mode("whisper")
            except OSError:
                log("! could not persist Whisper fallback; using it for this launch")
            speech_config = resolve_speech_config("auto", language=load_language_mode())
            model = speech_config.model_repo
            language_state["mode"] = language_mode(speech_config.language)
            use_nemotron = False
    if use_parakeet:
        log("checking local Parakeet model (first use downloads about 2.5 GB) ...")
        try:
            mlx_whisper = LocalParakeet(model)
            log(f"warming up {model} ...")
            warm_speech_runtime(adaptive=False, model=model, speech_config=speech_config, mlx_whisper=mlx_whisper)
            log("✓ Parakeet ready")
        except Exception as exc:
            # A missing or broken optional engine must never strand dictation.
            from speech_config import load_language_mode, save_engine_mode
            log(f"! Parakeet unavailable ({str(exc)[:120]}) — restoring Whisper")
            try:
                save_engine_mode("whisper")
            except OSError:
                log("! could not persist Whisper fallback; using it for this launch")
            mlx_whisper = None
            speech_config = resolve_speech_config("auto", language=load_language_mode())
            model = speech_config.model_repo
            language_state["mode"] = language_mode(speech_config.language)
            use_parakeet = False
    if not adaptive and not use_nemotron and not use_parakeet:
        import mlx_whisper as mlx_whisper_module
        log("checking local Whisper model (first launch downloads model files) ...")
        try:
            mlx_whisper = LocalWhisper(mlx_whisper_module, model)
        except ModelUnavailable as exc:
            log(f"✗ Whisper model unavailable: {exc}")
            sys.exit(1)
        import settings as automatic_settings
        from speech_config import automatic_languages
        # Automatic chooses only among the languages picked in Settings.
        mlx_whisper.allowed_languages = lambda: automatic_languages(
            automatic_settings.load(automatic_settings.SETTINGS_PATH)["languages"])
        mlx_whisper.speed = lambda: automatic_settings.load(automatic_settings.SETTINGS_PATH)["speed"]
        log(f"warming up {model} ...")
        warm_speech_runtime(adaptive=False, model=model, speech_config=speech_config, mlx_whisper=mlx_whisper)
        log("✓ model ready")

    active_engine = "nemotron" if use_nemotron else "parakeet" if use_parakeet else "whisper"

    # An unreadable history.jsonl (F16b) or learning set (N21) must not kill
    # the app before its menu bar exists. The adaptive lane's receipts depend
    # on both, so there it stays fatal.
    from history import HistoryStore
    from learning import LearningStore
    store = HistoryStore(tolerate_unreadable=not adaptive)
    learning_store = LearningStore(tolerate_unreadable=not adaptive)
    coordinator = history_coordinator(store, learning_store)
    if store.unreadable is not None:
        log(f"! History unreadable ({store.unreadable}): {history_unreadable_message(store)}")
    elif learning_store.unreadable is not None:
        log(f"! Learning set unreadable ({learning_store.unreadable}): "
            f"{learning_unreadable_message(learning_store)}")
    # N9b: each finished capture waits on disk until its History row exists;
    # what a crash, SIGKILL or logout left there becomes rows before the UI.
    from history import CaptureSpool
    spool = CaptureSpool(store.base_dir, log=log)
    spool_waiting = 0
    if spool.claim():
        spool_waiting = recover_capture_spool(spool, store, coordinator, model=model,
                                              speech_config=current_speech_config(),
                                              glossary_terms=glossary_terms, log=log)
    else:
        log("! another Sotto is using this data folder — captures stay in memory only until History has them")
    seed_totals(store)
    # Always available, no-model dependency guard: an earlier adaptive session
    # must remain revocation-safe even when this launch is non-adaptive.
    from adaptive_learning import AdaptiveLearning
    dependency_guard = AdaptiveLearning(store.base_dir)
    adaptive_runtime = None
    if adaptive:
        from maintainer.adaptive_runtime import AdaptiveRuntime
        adaptive_runtime = AdaptiveRuntime(history=store, learning=learning_store,
                                           coordinator=coordinator)
        adaptive_runtime.register_retained_history()
        adaptive_runtime.reconcile()
        # Receipt-gated exact-snapshot warmup; a missing baseline fails before
        # the daemon announces readiness and never falls back to repo/main.
        warm_speech_runtime(adaptive=True, model=model, speech_config=speech_config, adaptive_runtime=adaptive_runtime)
        log("✓ adaptive receipt-backed champion ready")

    status_ui = None
    if overlay:
        import ui
        status_ui = ui.init_app()
        status_ui.on_overlay_event = log
        status_ui.adaptive_mode = adaptive
        import settings as settings_store
        status_ui.automatic_languages = settings_store.load(settings_store.SETTINGS_PATH)["languages"] or None
        status_ui.set_engine_mode(active_engine,
                                  enabled=not adaptive and engine_switching)
        status_ui.set_language_mode(language_state["mode"],
                                    enabled=not adaptive and not use_nemotron and not use_parakeet)

    pending_deliveries = PendingDeliveries()
    finishing = PendingDeliveries()  # captures ended but not yet queued (see on_finish)
    ui_call, deliver_call = main_thread_dispatch(status_ui is not None, AppHelper.callAfter,
                                                 pending_deliveries)
    unreadable_alert = unreadable_store_alert(store, learning_store)
    if status_ui is not None and unreadable_alert is not None:
        ui_call(status_ui.show_error, unreadable_alert[0], unreadable_alert[1] + spool_waiting_note(spool_waiting))

    capture = CaptureService()
    capture.restart_callback = restart.request
    capture.idle_release_s = idle_release
    capture_gate = CaptureGate()
    if status_ui is not None:
        status_ui.level_source = lambda: capture.latest_rms
        status_ui.live_text_source = (
            (lambda: live_preview(capture.live_text())) if use_nemotron else None)

    # Single-flight transcription: one prewarmed worker, strict FIFO — live
    # dictations and retries can never run the model (or the clipboard cycle)
    # concurrently, and results arrive in capture order.
    jobs: queue.Queue = queue.Queue()
    model_activity = {"last_finished": time.monotonic(), "rewarming": False}

    # Gesture callbacks mark the capture boundary. The audio engine starts and
    # stops off-thread; the ASR model stays resident independently.
    def on_start() -> None:
        with capture_gate.starting() as allowed:
            if shutdown.requested():
                return
            if allowed:
                stream = None
                if use_nemotron:
                    from streaming_audio import StreamingCapture
                    stream = StreamingCapture(nemotron)
                cold = capture.begin(stream=stream, enqueue=lambda job: shutdown.enqueue(jobs, job))
        if not allowed:
            # Quit, Restart, an update or an engine switch is draining (N3,
            # the Mac side of Windows F70): the mic stays closed, and the
            # gesture ends too, as on_mic_failed does, or it would believe it
            # is recording and a hands-free orb would stay up with no mic.
            log(f"○ Sotto is {lifecycle.closed or 'restarting'} — this press is ignored")
            hands_free = engine.snapshot()[1]  # a menu start: its orb shows next
            engine.force_finish(finish=hide_refused_start if hands_free else (lambda: None))
            return
        now = time.monotonic()
        if model_rewarm_due(model_activity["last_finished"], now,
                            rewarming=bool(model_activity["rewarming"]),
                            queue_idle=jobs.empty(), adaptive=adaptive or use_nemotron):
            model_activity["rewarming"] = True
            if not shutdown.enqueue(jobs, ("warmup",)):
                model_activity["rewarming"] = False
        if status_ui:
            ui_call(status_ui.show_recording)
        if not cold:
            log("● recording")
            return
        # Cold start after idle: audio does not exist until the engine is
        # live (measured up to ~5s on this machine). Say so, and flip the
        # indicator to "live" the moment the first block lands.
        log("● waking the mic — speak when the orb stops blinking")
        if status_ui:
            ui_call(status_ui.show_waking)

            def announce_live() -> None:
                deadline = time.monotonic() + 10
                while capture.is_waking() and time.monotonic() < deadline:
                    time.sleep(0.05)
                # A start that failed every try also ends the wait: say
                # "live" only when an engine actually exists.
                if engine.snapshot()[0] and capture.is_live():
                    ui_call(status_ui.show_recording)
                    log("● recording (mic live)")

            threading.Thread(target=announce_live, daemon=True).start()

    def on_finish() -> None:
        # In flight from before the capture ends until its job is queued, so
        # Quit's drain never sees an idle gap between the two; ``finishing``
        # lets the teardown wait for that audio too (N9).
        pending_deliveries.add()
        finishing.add()
        try:
            finish_capture()
        finally:
            finishing.finish()
            pending_deliveries.finish()

    def finish_capture() -> None:
        released_at = time.monotonic()  # latency is measured from the key release
        if shutdown.requested():
            shutdown.stop_capture(capture)
            return
        captured_ts = time.time()
        raw, stream = capture.end(include_stream=True)
        capture.release_soon()  # audio is in hand — free the mic now
        seconds = len(raw) / max(capture.native_rate, 1.0)
        log(f"○ {seconds:.2f}s captured")
        if seconds < 0.25:
            if stream is not None:
                stream.cancelled.set()
                shutdown.enqueue(jobs, ("stream-close", stream))
            if status_ui:
                ui_call(status_ui.hide)
            return
        if status_ui:
            ui_call(status_ui.show_transcribing)
        capture_id = uuid.uuid4().hex
        # On disk until its History row exists (N9b); the write runs on the
        # spool's own thread, never on the event tap.
        spool.submit(capture_id, raw, capture.native_rate, captured_ts)
        # Shutdown may come while the audio is in hand: keep it in History (N9).
        shutdown.enqueue(jobs, ("live", raw, capture.native_rate, captured_ts,
                                released_at, capture_id, current_speech_config(), stream),
                         keep=worker.keep_untranscribed)

    def on_discard() -> None:
        capture.abort()
        capture.release_soon()
        if status_ui:
            ui_call(status_ui.hide)
        log("○ tap ignored")

    def on_hands_free() -> None:
        if status_ui:
            ui_call(status_ui.show_hands_free, hands_free_hint(binding["trigger"]))

    def hide_refused_start() -> None:
        """The finish of a hands-free start the closed gate refused: there is
        no capture, so only its orb goes away. A refused key press shows no
        orb, and hiding then would hide a draining dictation's one."""
        if status_ui:
            ui_call(status_ui.hide)

    def on_mic_failed() -> None:
        """The engine never started under this capture (device busy or gone).
        End the gesture so the key-up finds nothing to finish, and say so
        once — a dead mic used to show "recording" and then lose the dictation
        without a word. Finishing (not discarding) keeps whatever a
        mid-capture rebuild had already recorded."""
        if engine.force_finish():
            log("✗ could not start the microphone — dictation cancelled")
        if status_ui:
            ui_call(status_ui.show_error, "Could not start the microphone",
                    "Sotto could not open the input device. Check that another app "
                    "is not holding it and that Sotto may use the microphone "
                    "(System Settings → Privacy & Security → Microphone), then "
                    "press the key again.")

    engine = GestureEngine(on_start, on_finish, on_discard, on_hands_free=on_hands_free)
    capture.on_start_failed = on_mic_failed
    lifecycle = Lifecycle(
        shutdown, capture_gate, end_recording=lambda: end_capture_now(engine, capture, on_finish),
        busy=lambda: dictation_in_flight(capture, jobs, pending_deliveries,
                                         warming=bool(model_activity["rewarming"])))
    # Real key-downs seen by the tap (Sotto's tagged events excluded): typing
    # after a paste means "scratch that" must not undo it.
    user_keydowns = {"count": 0}
    # The event tap (trigger edges, the hotkey, the resync poller); it starts
    # below, once the worker runs. Quartz is its adapter.
    hotkeys = HotkeyListener(
        trigger, hotkey_spec, quartz=Quartz, engine=engine, capture=capture, shutdown=shutdown,
        is_own_event=is_own_event, user_keydowns=user_keydowns, log=log, status_ui=status_ui,
        ui_call=ui_call, clock=time.monotonic, sleep=time.sleep)
    # The listener's mutable binding: Settings moves the trigger at the next press.
    binding = hotkeys.binding

    def insert_text(text: str) -> bool:
        import settings
        preferences = settings.load(settings.SETTINGS_PATH)
        return inject(text, insert_mode=preferences["insert_mode"], spacing=preferences["spacing"])

    def delivery_note(title: str, message: str) -> None:
        if status_ui is not None:
            ui_call(status_ui.show_info, title, message)

    # Finished text and "scratch that" share one FIFO (see DeliveryQueue):
    # capture order is kept, and nothing is posted while a key is held.
    delivery = DeliveryQueue(
        insert=insert_text, undo_keys=lambda: post_command_key(6), keys_held=hotkeys.keys_held,
        recording=lambda: engine.snapshot()[0], frontmost_pid=frontmost_pid,
        keydowns=lambda: user_keydowns["count"], secure_input=secure_input_active,
        call_after=AppHelper.callAfter, note=delivery_note, shutdown_requested=shutdown.requested,
        on_done=pending_deliveries.finish, own_window_front=own_window_without_text_field,
        copy=set_clipboard_text)

    # deliver_call counts each of these in pending_deliveries when it schedules
    # it; the queue's on_done finishes that entry once the item is delivered,
    # dropped or cleared, so Quit and the update restart wait for the paste.
    def inject_when_clear(text: str, _attempts: int = 0, in_history: bool = True) -> None:
        """Queue a finished dictation for the cursor (worker call shape kept)."""
        delivery.paste(text, in_history)

    def undo_when_clear(_attempts: int = 0) -> None:
        """Queue a 'scratch that' behind whatever is still waiting to paste."""
        delivery.undo()

    # The menu bar's actions; the worker uses its History refresh and copy.
    controller = AppController(
        shutdown=shutdown, lifecycle=lifecycle, jobs=jobs, store=store, coordinator=coordinator,
        dependency_guard=dependency_guard, adaptive_runtime=adaptive_runtime, status_ui=status_ui,
        ui_call=ui_call, deliver_call=deliver_call, pending_deliveries=pending_deliveries,
        set_clipboard=set_clipboard_text, current_speech_config=current_speech_config, log=log,
        history_keep=HISTORY_KEEP, overlay_learning_state=overlay_learning_state,
        active_learning_history_ids=_active_learning_history_ids, nonadaptive_correct=nonadaptive_correct,
        nonadaptive_delete=nonadaptive_delete, nonadaptive_clear=nonadaptive_clear,
        nonadaptive_revoke_learning=nonadaptive_revoke_learning, save_audio_copy=save_audio_copy,
        adaptive=adaptive, use_nemotron=use_nemotron, use_parakeet=use_parakeet,
        engine_switching=engine_switching, active_engine=active_engine, settings_locked=settings_locked,
        binding=binding, rebind=hotkeys.rebind, restart=restart, language_lock=language_lock,
        language_state=language_state, save_language_mode=save_language_mode, parakeet_cached=parakeet_cached,
        persist_engine_and_restart=persist_engine_and_restart, run_installer=run_installer, triggers=TRIGGERS,
        trigger_labels=TRIGGER_LABELS, sotto_script=Path(__file__).resolve(), engine=engine, capture=capture,
        on_finish=on_finish)

    # Built here, after inject_when_clear / undo_when_clear exist: the worker
    # holds its collaborators instead of resolving them late like the closure
    # did. Imported inside run() so transcription.py can bind sotto's helpers
    # at import time without a cycle at module load.
    from transcription import TranscriptionWorker
    worker = TranscriptionWorker(
        shutdown=shutdown, jobs=jobs, log=log, model=model, mlx_whisper=mlx_whisper,
        nemotron=nemotron, current_speech_config=current_speech_config,
        coordinator=coordinator, refresh_history=controller.refresh_history,
        adaptive_runtime=adaptive_runtime, adaptive=adaptive, use_nemotron=use_nemotron,
        status_ui=status_ui, ui_call=ui_call, deliver_call=deliver_call,
        inject_when_clear=inject_when_clear,
        undo_when_clear=undo_when_clear, copy_text=controller.copy_text, record_totals=record_totals,
        model_activity=model_activity, glossary_terms=glossary_terms, spool=spool)
    transcription_thread = threading.Thread(target=worker.run, daemon=True)
    transcription_thread.start()

    hotkeys.start()

    restart_drain = {"deadline": None, "kept": False}

    def keep_untranscribed() -> None:
        """A capture still untranscribed at teardown stays in History with its
        audio (N9), for Retry: every queued live capture, the one the worker
        is on (whose History append, if already under way, gets
        APP_DRAIN_TIMEOUT to finish) and any it gave up on because shutdown
        came first. The worker settles each capture once its append succeeds,
        so one it saved is never kept again, and a model that returns after
        this keep saves nothing (shutdown already fences its result).

        A finish still holding its audio short of the queue gets
        APP_DRAIN_TIMEOUT to reach it (a refused enqueue keeps it), and the
        worker the same, after the running capture is kept, to file a capture
        it took just as shutdown came. After SIGTERM these waits share one
        deadline (ShutdownBoundary.teardown_wait_s), so launchd's SIGKILL
        never comes before the keeps.

        Side effects: History rows; may wait and join the worker briefly; logs."""
        wait_until_idle(lambda: finishing.count() > 0, poll_s=0.02,
                        deadline_s=shutdown.teardown_wait_s(APP_DRAIN_TIMEOUT, time.monotonic()))
        shutdown.discard_queued(jobs, keep=worker.keep_untranscribed)
        running = worker.current_job
        if running is not None and running[0] == "live":
            worker.keep_untranscribed(
                running, wait_s=shutdown.teardown_wait_s(APP_DRAIN_TIMEOUT, time.monotonic()))
        transcription_thread.join(shutdown.teardown_wait_s(APP_DRAIN_TIMEOUT, time.monotonic()))
        for job in list(worker.abandoned):
            worker.keep_untranscribed(job)

    def stop_runtime_on_main() -> None:
        """Main-loop teardown once shutdown is requested (Quit and the signal
        drain have already waited for the dictation in flight).

        Side effects: the first pass puts back a pending clipboard restore
        (N6) and keeps untranscribed captures in History (N9); then ends the
        process — exec for a foreground restart, exit status
        RESTART_FAILED_EXIT if that exec fails (N27)."""
        shutdown.stop_capture(capture)
        if not restart_drain["kept"]:
            restart_drain["kept"] = True
            flush_clipboard_restore(shutdown.teardown_wait_s(
                SIGTERM_RESTORE_WAIT_S if shutdown.stop_signal == signal.SIGTERM else RESTORE_DELAY_S,
                time.monotonic()))
            keep_untranscribed()
        shutdown.discard_queued(jobs)
        drained = True
        if restart.foreground_pending or nemotron is not None:
            # Let the worker finish its current job; shutdown already fences
            # every paste/commit, so nothing it produces is kept.
            transcription_thread.join(APP_DRAIN_TIMEOUT)
            drained = not transcription_thread.is_alive()
        if not drained:
            if restart_drain["deadline"] is None:
                restart_drain["deadline"] = time.monotonic() + RESTART_DRAIN_DEADLINE_S
            if time.monotonic() < restart_drain["deadline"]:
                AppHelper.callLater(0.1, stop_runtime_on_main)
                return
            if not restart.foreground_pending:
                # A wedged native call must not hang Quit forever, and freeing
                # the native engine underneath it could crash: leave directly.
                log("! transcription did not finish in time — quitting anyway")
                os._exit(0)
            # exec replaces the whole image, so a wedged native call cannot
            # follow it; waiting forever would strand the user instead.
            log("! transcription did not finish in time — restarting anyway")
        if nemotron is not None and drained:
            # Cocoa terminate exits without unwinding run()'s finally block.
            # Free native streams/model before Metal's global destructors run.
            nemotron.close()
        try:
            Quartz.CGEventTapEnable(hotkeys.tap, False)
        except Exception:
            pass
        if restart.foreground_pending:
            if not capture.wait_released(RESTART_RELEASE_WAIT_S):
                log("! microphone teardown still running — restarting anyway")
            try:
                restart.exec_foreground()
            except Exception as exc:
                # Shutdown is one-way, so this process cannot carry on. Exit
                # non-zero: KeepAlive (the login item, the LaunchAgent) starts
                # Sotto again; exit 0 would be a Quit it leaves alone, and an
                # alert queued now would never be shown.
                restart.foreground_pending = False
                try:
                    restart.failed(f"foreground restart failed ({type(exc).__name__}: {str(exc)[:120]})")
                except Exception as callback_exc:
                    log(f"! restart failure handling failed: {str(callback_exc)[:120]}")
                log(f"! exiting with status {RESTART_FAILED_EXIT} so launchd can start Sotto again; "
                    "from a terminal, run the command again")
                os._exit(RESTART_FAILED_EXIT)
        if status_ui is not None:
            ui_call(status_ui.hide)
            try:
                import AppKit
                AppKit.NSApp.terminate_(None)
            except Exception:
                pass
        else:
            try:
                Quartz.CFRunLoopStop(Quartz.CFRunLoopGetMain())
            except Exception:
                pass

    def shutdown_watcher() -> None:
        shutdown.event.wait()
        try:
            AppHelper.callAfter(stop_runtime_on_main)
        except Exception:
            # The handler remains event-only; a headless/embedded caller still
            # gets capture and queue fencing even if AppKit is unavailable.
            shutdown.stop_capture(capture)
            shutdown.discard_queued(jobs)

    threading.Thread(target=shutdown_watcher, daemon=True).start()
    threading.Thread(target=lifecycle.watch_signals, daemon=True, name="sotto-signals").start()

    if status_ui is not None:
        # menu actions + system lifecycle hooks
        status_ui.set_history_callbacks(controller.menu_callbacks())
        status_ui.on_wake = lambda: (not shutdown.requested() and
                                     (engine.force_reset(), Quartz.CGEventTapEnable(hotkeys.tap, True)))
        controller.refresh_history()

    log(f"listening on {trigger} · hold to talk, double-tap for hands-free"
        + (f" · {hotkey_spec}: hold or tap (remote-friendly)" if hotkeys.hotkey else "")
        + f" · model: {model.split('/')[-1]} · language: {language_state['mode']} · ^C to quit")
    try:
        with shutdown_signal_handlers(shutdown):
            # Cocoa/CFRunLoop can otherwise sit in native code indefinitely;
            # CPython only delivers SIGINT/SIGTERM on a main-thread bytecode
            # boundary. A lightweight timer gives those handlers a boundary.
            def signal_tick():
                if not shutdown.requested():
                    AppHelper.callLater(0.25, signal_tick)

            AppHelper.callLater(0.25, signal_tick)
            if status_ui is not None:
                import ui
                ui.run_loop()
            else:
                Quartz.CFRunLoopRun()
    except KeyboardInterrupt:
        shutdown.request()
        log("\nshutting down")
    finally:
        shutdown.stop_capture(capture)
        shutdown.discard_queued(jobs)
        transcription_thread.join(APP_DRAIN_TIMEOUT)
        if nemotron is not None and not transcription_thread.is_alive():
            nemotron.close()


def doctor() -> None:
    import subprocess
    import ApplicationServices

    trusted = ApplicationServices.AXIsProcessTrusted()
    log(("✓" if trusted else "✗") + " accessibility (this terminal)")
    microphone = microphone_permission(request=True)
    log(("✓" if microphone else "✗") + " microphone permission (terminal/Python)")
    if not microphone:
        log("→ System Settings → Privacy & Security → Microphone → enable your terminal/Python")
    usable_input = False
    # Probe the input device in a subprocess: CoreAudio hard-crashes (a native
    # SIGSEGV, not a catchable exception) in sessions with no usable audio
    # context, and a diagnostic must survive the conditions it diagnoses.
    # The probe is doctor_input_probe: the capture path's own pin + input bus.
    probe = "import sotto; raise SystemExit(sotto.doctor_input_probe())"
    try:
        result = subprocess.run([sys.executable, "-c", probe],
                                capture_output=True, text=True, timeout=30,
                                cwd=str(Path(__file__).resolve().parent))
    except subprocess.TimeoutExpired:
        log("✗ input device: probe timed out (waiting on a permission prompt?)")
    else:
        if result.returncode == 0:
            usable_input = True
            mics = [line.strip()[len("mic:"):].strip()
                    for line in result.stderr.splitlines()
                    if line.strip().startswith("mic:")]
            log(f"✓ input device: {result.stdout.strip()} "
                f"({mics[0] if mics else 'AVAudioEngine'})")
        else:
            log("✗ input device: no usable audio input in this session "
                "(microphone permission missing, or headless/SSH context)")
    if not trusted:
        log("→ System Settings → Privacy & Security → Accessibility → add your terminal")
    if not (trusted and microphone and usable_input):
        sys.exit(1)


# The CLI commands that run the maintainer lane (they need maintainer/).
LANE_COMMANDS = frozenset({"learning-preflight", "learning-review-list", "learning-evaluate", "learning-rollback"})


def maintainer_lane_available() -> bool:
    """Whether this copy includes the maintainer lane; the alpha source
    archive leaves the ``maintainer`` package out. Side effects: imports the
    package's (empty) ``__init__`` only."""
    import importlib.util
    try:
        return importlib.util.find_spec("maintainer.adaptive_runtime") is not None
    except ImportError:
        return False


def main() -> None:
    from speech_config import SUPPORTED_LANGUAGES
    parser = argparse.ArgumentParser(description="Local push-to-talk dictation.")
    parser.add_argument("command", nargs="?", default="run",
                        choices=["run", "doctor", "setup", "dictionary", "learning-status", "learning-list",
                                 "learning-remove", "export-learning", "learning-preflight",
                                 "learning-review-list", "learning-evaluate", "learning-rollback"])
    parser.add_argument("--trigger", default=None, choices=sorted(TRIGGERS),
                        help="dictation key for this run (default: Settings, initially right-option)")
    parser.add_argument("--profile", default=None,
                        choices=("auto", "nemotron-en", "parakeet"),
                        help="local speech profile (default: saved engine, initially auto)")
    parser.add_argument("--model", default=None,
                        help="explicit local MLX model repository override")
    parser.add_argument("--language", choices=SUPPORTED_LANGUAGES, default=None, metavar="CODE",
                        help="explicit language override; default uses the profile")
    parser.add_argument("--glossary", help="local UTF-8 text or JSON glossary")
    parser.add_argument("--adaptive", action="store_true",
                        help="enable human-grounded adaptive English runtime (forces --language en)")
    parser.add_argument("--output", help="empty directory for export-learning")
    parser.add_argument("--sample-id", help="active learning sample ID for learning-remove")
    parser.add_argument("--no-overlay", action="store_true",
                        help="headless: no pill, no menu-bar icon")
    parser.add_argument("--hotkey", default=None,
                        help="hotkey that works over remote desktop — hold it "
                             "or tap it, e.g. ctrl-opt-d or f13 ('' to disable)")
    parser.add_argument("--idle-release", type=float, default=IDLE_STOP_S,
                        help="seconds to retain the mic after key-up "
                             "(0 = immediately; negative = never)")
    args = parser.parse_args()
    if (args.command in LANE_COMMANDS or (args.command == "run" and args.adaptive)) and not maintainer_lane_available():
        parser.error("this needs the maintainer lane (maintainer/), which this copy of Sotto does not include")
    if app_mode() and args.command == "run":
        from sotto_paths import DATA_DIR
        route_app_logs(DATA_DIR)

    if args.command == "doctor":
        doctor()
        return
    if args.command == "setup":
        # Download the default model now (with progress) so the first
        # dictation, or the first Sotto.app launch, is ready immediately.
        try:
            if args.profile == "parakeet":
                from speech_config import MODEL_PROFILES
                engine, path = "Parakeet", LocalParakeet(MODEL_PROFILES["parakeet"].repo).path
            else:
                engine, path = "Whisper", LocalWhisper(None, DEFAULT_MODEL).path
        except ModelUnavailable as exc:
            print(f"✗ {exc}")
            sys.exit(1)
        print(f"✓ {engine} model ready ({Path(path).name[:12]}) in {SOTTO_HF_HOME}")
        from offline_runtime import offline_requested
        import vad
        try:
            ready = vad.install(allow_download=not offline_requested())
            print(f"✓ voice detection ready (Silero VAD v6.2) in {ready.parent}")
        except Exception as exc:  # advisory: transcription works without it
            print(f"! voice detection not installed ({str(exc)[:120]}); run setup again later")
        return
    if args.command == "dictionary":
        import dictionary
        rules = dictionary.load(dictionary.DICTIONARY_PATH)
        print(f"{dictionary.DICTIONARY_PATH} · {len(rules)} rule(s)")
        for rule in rules:
            print(f"  {rule.heard} {dictionary.SEPARATOR} {rule.write}")
        return

    # Corpus administration must remain useful on an offline Mac with no MLX
    # runtime installed and must never trigger an inference model download.
    if args.command in {"learning-status", "learning-list", "learning-remove", "export-learning"}:
        from learning import LearningStore
        if args.command == "learning-status":
            print(adaptive_status_text(LearningStore.admin_snapshot()))
            return
        if args.command == "learning-list":
            print(learning_list_text(LearningStore.admin_snapshot()))
            return
        if args.command == "learning-remove":
            if not args.sample_id:
                parser.error("learning-remove requires --sample-id SAMPLE_ID")
            from adaptive_learning import AdaptiveLearning
            from history import STORE_DIR
            if not remove_learning_sample(
                    args.sample_id, history_id_lookup=LearningStore.admin_history_id,
                    adaptive_revoke=lambda ident: AdaptiveLearning(STORE_DIR).revoke(ident, reason="learning_sample_removed"),
                    artifact_revoke=LearningStore.admin_revoke):
                parser.error("Learning sample was not found or was already removed")
            print(f"Removed learning sample {args.sample_id}")
            return
        if not args.output:
            parser.error("export-learning requires --output DIRECTORY")
        try:
            count = export_learning_snapshot(args.output)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        print(f"Exported {count} learning item(s) to {Path(args.output).expanduser()}")
        return

    if args.command in {"learning-preflight", "learning-review-list", "learning-evaluate", "learning-rollback"}:
        from maintainer.adaptive_runtime import AdaptiveRuntime
        runtime = AdaptiveRuntime()
        if args.command == "learning-preflight":
            for result in runtime.preflight():
                print(f"backend={result['backend']} stable_id={result['stable_id']} success=true")
            return
        if args.command == "learning-review-list":
            for row in runtime.review_list():
                fields = (f"history_id={row['history_id']} ts={row['captured_ts']} status={row['status']}"
                          + (f" reason={row.get('skip_reason')}" if row.get("skip_reason") else ""))
                print(fields)
            return
        if args.command == "learning-evaluate":
            decision = runtime.evaluate()
            print("decision=" + str(decision.get("outcome")) + (f" winner={decision.get('winner')}" if decision.get("winner") else ""))
            return
        result = runtime.rollback()
        print(f"rollback={result['outcome']}")
        return

    try:
        from speech_config import load_engine_mode, load_glossary, load_language_mode, resolve_speech_config
        saved_engine = load_engine_mode() if not args.adaptive and args.model is None else "whisper"
        profile = args.profile or {"nemotron": "nemotron-en", "parakeet": "parakeet"}.get(saved_engine, "auto")
        if args.adaptive and profile in ("nemotron-en", "parakeet"):
            parser.error("Nemotron and Parakeet cannot be combined with --adaptive")
        if args.adaptive and args.language not in {None, "en"}:
            parser.error("--adaptive requires English; use --language en")
        language = args.language
        if not args.adaptive and language is None and profile == "auto":
            language = load_language_mode()
        speech_config = resolve_speech_config(profile, model=args.model,
                                              language="en" if args.adaptive else language)
        source = glossary_source(args.glossary)
        glossary_terms = load_glossary(source) if source else ()
    except ValueError as exc:
        parser.error(str(exc))
    import settings
    preferences = settings.load(settings.SETTINGS_PATH)
    trigger = args.trigger or preferences["trigger"]
    if trigger not in TRIGGERS:
        trigger = settings.DEFAULTS["trigger"]
    run(trigger, overlay=not args.no_overlay,
        hotkey_spec=args.hotkey if args.hotkey is not None else preferences["hotkey"],
        settings_locked={name for name, flag in (("trigger", args.trigger), ("hotkey", args.hotkey))
                         if flag is not None},
        idle_release=args.idle_release,
        speech_config=speech_config, glossary_terms=glossary_terms, adaptive=args.adaptive,
        engine_switching=args.profile is None and args.model is None)


if __name__ == "__main__":
    main()
