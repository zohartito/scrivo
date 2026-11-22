# Private own-voice benchmark

No voice measurements accompany this tooling. It compares **Apple's Speech
framework (on-device)** with `mlx-community/whisper-large-v3-turbo`, Scrivo's
default `auto` model, on the same files. Add `--with-parakeet` to either aggregate
runner to also measure the pinned Parakeet TDT v3 model from the private cache.
The default remains Apple + Whisper; no Parakeet accuracy results accompany
this option. `benchmark.py` supports MLX Whisper repositories only: a
`qwen3asr=...` label does not add a Qwen backend. Qwen3-ASR and Nemotron are not
benchmarked here. Nemotron's native installation and streaming lifecycle need
separate benchmark validation; no adapter was added to that CLI.

## Prepare once (Apple Silicon macOS)

Use a framework build of Python 3.12 (for example Homebrew) and `ffmpeg` on
`PATH`. From `/path/to/repo`, use a separate environment; leave the app and its venv alone:

```sh
python3.12 -m venv .venv-bench
.venv-bench/bin/python -m pip install pip==26.2.1
.venv-bench/bin/python -m pip install -r requirements-alpha.txt pyobjc-framework-Speech==12.2.2
.venv-bench/bin/python scripts/bench_host.py
mkdir -p "$HOME/ScrivoBench"
chmod 700 "$HOME/ScrivoBench"
```

The host is a copy of the Python interpreter with Speech/Microphone purpose
strings, inside the ignored `.venv-bench` only. It requests no permissions during
setup. The recording and recognition commands use this host automatically;
allow Microphone/Speech Recognition when macOS asks. No privacy settings are
changed by the scripts. If permission is denied, allow it manually and rerun.
If on-device recognition fails, stop after at most two manual attempts; there is
no server fallback or automatic retry.

Separately download **only the pinned model**, before the private recognition run:

```sh
.venv-bench/bin/python -c 'from pathlib import Path; from huggingface_hub import snapshot_download; from speech_config import MODEL_PROFILES, MODEL_REVISIONS; repo = MODEL_PROFILES["auto"].repo; snapshot_download(repo, revision=MODEL_REVISIONS[repo], cache_dir=str(Path.home()/"ScrivoBench/models"))'
```

For the optional Parakeet comparison, separately provision its pinned snapshot
in the **same benchmark cache**: use `MODEL_PROFILES["parakeet"].repo` in the
setup command above instead of `MODEL_PROFILES["auto"].repo`. Both snapshots
are needed. Parakeet requires `config.json` and `model.safetensors` at its pinned
revision; a missing or partial snapshot stops the run, with no download fallback.
The alpha requirements already include `parakeet-mlx`.

The benchmark only loads that cache, offline. It does not read Scrivo's History,
dictionary, settings or model cache. Do not run `benchmark.py`'s CLI for this
workflow: its JSON contains per-clip private content.

## Record 10 clips first; expand to 30

Choose your own emails, studio notes and technical sentences. Do not retrieve
saved dictation History. Keep one speaker, microphone, mic distance and room;
write those conditions in a private note. Mix ordinary prose with names,
numbers and terms you actually use (for example Make2D, poché, Dify, pgvector,
Qwen3 or A6000). Decide the clips before seeing either recognizer's output.
You supply all spoken text and exact references; there are no supplied scripts
or generated speech.

```sh
.venv-bench/bin/python scripts/record_bench_clip.py --id c01 --dir "$HOME/ScrivoBench" --seconds 15
```

Press Return, wait for **Speak now**, then speak. The recorder stops after the
chosen duration (1–30 seconds), saves `audio/c01.wav` as 16 kHz mono PCM16,
and appends a row to `manifest.jsonl`. It uses Scrivo's microphone selection
(worn headset when available, otherwise built-in); use the same setup each time.
Repeat with `c02` through `c10`, later through `c30`. Existing takes/IDs are
refused. On an interrupted save, inspect the private folder before retrying.

In `ScrivoBench/manifest.jsonl`, fill each `reference` with **exactly what you
said**, including mistakes and fillers, after listening to the WAV. Fill
`required_terms` with a JSON list of words/phrases actually spoken in that clip;
leave `[]` when there are none. `tags` are optional and never reported. Leave
`language` as `en` for English. For another language, use the recorder's
`--language` and the runner's matching `--locale` (for example `fr` / `fr-FR`).
Use one language per run. Never use a recognizer output as the reference.

## Produce both reports with one command

After all 10–30 references are complete:

```sh
.venv-bench/bin/python scripts/own_voice_bench.py "$HOME/ScrivoBench/manifest.jsonl" --out "$HOME/ScrivoBench/results/own-voice.md" --platform-out "$HOME/ScrivoBench/results/platform-baseline.md"
```

Add `--dry-run` to validate first without recognition, permissions or output
files. Use new output names for each real run; existing files are never
overwritten. `--locale en-US` is the default. Both reports come from one
recognition pass, and the console prints only the platform aggregate table.
For just that table, use `scripts/platform_baseline.py MANIFEST --out REPORT.md`.
Reports stay private until reviewed; no README or public results are updated.

The platform table has exactly `backend, n, wer, required_term_hit_rate,
median_latency_s, date, macos_version`. The own-voice report adds CER, paired
clip-bootstrap 95% intervals, a paired WER difference, model revision and method
limits. With `--with-parakeet`, both reports gain a third backend row, and the
own-voice report adds Parakeet minus Whisper WER with a paired 95% interval
(negative favors Parakeet). Every backend uses the same resampled clip indices.
An interval including zero does not resolve a difference. WER/CER are micro
averages; terms use total matches/total annotated terms, or `n/a` with no terms.
No paths, IDs, tags, references, hypotheses or
failure examples are retained in either report. Backend stdout/stderr and raw
exceptions are suppressed, including native console output.

### Optional literal numeric-form audit

Add `--numeric-form-audit` to **`own_voice_bench.py` only** to append a table of
`backend, matched_clips, eligible_clips`. A clip is eligible when its reference
contains a digit-written expression; it matches only when the entire ordered
expression list equals the hypothesis's list. Extra, missing, reordered and
repeated expressions matter. Counts are per clip, not per expression; with no
eligible clips the table shows `n/a` matched and `0` eligible. This also works
with `--with-parakeet`; `--dry-run` still does no scoring.

The pure helper in `evaluation.py` greedily extracts digit runs, dot decimals
(including `.5`), slash fractions, space/tab mixed numbers and colon-separated
runs, retaining an adjacent `+`/`-` sign. It preserves digits, separators and
internal whitespace literally; it ignores surrounding prose/punctuation. Thus
`1/8`, `1.8` and `1 1/8` are distinct, as are `1:50` and `1.50`. This is a
**literal-form audit**, not semantic equivalence: `1.50` differs from `1.5`, and
spoken number words are not interpreted. A reference with only number words
is ineligible even if the hypothesis uses digits. This is not a general numeric
parser (no locale-aware comma decimals, scientific notation or Unicode fraction
conversion); digit runs inside identifiers are included too.

Only aggregate matched/eligible counts survive scoring, never expression lists
or per-clip audit results. No new intervals or promotion thresholds are added.
Without the flag the report bytes stay unchanged; with or without it the
platform schema, WER/CER, term metrics and adaptive promotion rules are unchanged.
Deterministic stub checks exercise the real scoring/report path: six altered
eligible pairs are rejected, including three punctuation changes with zero
legacy WER/CER; two literal-form controls match and two references without digits
are unscored. These checks are not own-voice recognition measurements.

Apple runs first; Whisper runs second; optional Parakeet runs third, with no
warmup. Latency includes file loading and the first decode's cold start, but
excludes snapshot resolution, authorization, scoring and capture. All receive
the same untrimmed WAV with no glossary/dictionary hints. Apple and Whisper
use a fixed language; Parakeet detects language itself (25 European languages),
with no locale hint. Its adapter follows `LocalParakeet`'s bfloat16 load and
`get_logmel`/`generate` recipe on float arrays decoded from the 16 kHz PCM16
WAV; it never imports the app. This tests the **models**, not Scrivo's full
delivery/cleanup pipeline. One speaker and 10–30 selected clips cannot establish
general accuracy; intervals resample clips, not speakers. Backend order and
resident models can affect memory use and latency; this is not an isolated
engine speed test. A real quality comparison still requires your completed
corpus and an Apple Silicon Mac; stub checks are not recognition measurements.

[Apple's capability check](https://developer.apple.com/documentation/speech/sfspeechrecognizer/supportsondevicerecognition)
and [on-device request flag](https://developer.apple.com/documentation/speech/sfspeechrecognitionrequest/requiresondevicerecognition)
are both enforced. Apple describes shared recognition models in
[DictationTranscriber](https://developer.apple.com/documentation/speech/dictationtranscriber),
but this does not test the complete macOS Dictation product.

Stub-only checks (no microphone, audio synthesis, model or permissions):

```sh
.venv-bench/bin/python scripts/test_alpha.py -p 'test_*bench*.py'
.venv-bench/bin/python scripts/test_alpha.py -p test_platform_baseline.py
```
