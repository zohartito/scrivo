"""One private corpus, one recognition pass, two aggregate-only reports."""
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import bench_common as bench  # noqa: E402
from scripts.bench_host import ensure_host  # noqa: E402

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 376


def statistics_for(result, indices):
    samples = [result.samples[index] for index in indices]
    words = sum(row.reference_words for row in samples)
    characters = sum(row.reference_characters for row in samples)
    terms = sum(row.required_term_count for row in samples)
    return (
        sum(row.word_distance for row in samples) / words,
        sum(row.character_distance for row in samples) / characters,
        sum(row.matched_required_terms for row in samples) / terms if terms else None,
        statistics.median(result.latencies[index] for index in indices),
    )


def intervals(measurements, *, replicates=BOOTSTRAP_REPLICATES):
    """Paired clip resampling, percentile 95% CIs; no normal approximation.

    Use the same draw for every backend. Terms have no interval if any draw has
    no required terms, rather than silently conditioning on term-bearing draws.
    Return metric CIs and WER-difference CIs keyed by (candidate, baseline) index:
    Whisper minus Apple, then each optional backend minus Whisper.
    """
    if len(measurements) < 2:
        raise ValueError("Bootstrap requires at least two paired backends.")
    n = len(measurements[0].samples)
    if n == 0 or replicates < 2 or any(len(row.samples) != n or len(row.latencies) != n for row in measurements):
        raise ValueError("Bootstrap requires equal nonempty paired samples and at least two draws.")
    rng = random.Random(BOOTSTRAP_SEED)
    draws = [[[] for _ in range(4)] for _ in measurements]
    pairs = [(1, 0)] + [(index, 1) for index in range(2, len(measurements))]
    differences = {pair: [] for pair in pairs}
    for _ in range(replicates):
        indices = [rng.randrange(n) for _ in range(n)]
        values = [statistics_for(result, indices) for result in measurements]
        for model, metrics in zip(draws, values, strict=True):
            for metric, value in zip(model, metrics, strict=True):
                metric.append(value)
        for (candidate, baseline), paired in differences.items():
            paired.append(values[candidate][0] - values[baseline][0])

    def percentile(values):
        if any(value is None for value in values):
            return None
        ordered = sorted(values)
        return ordered[int(.025 * (replicates - 1))], ordered[int(.975 * (replicates - 1))]

    return ([[percentile(metric) for metric in model] for model in draws],
            {pair: percentile(values) for pair, values in differences.items()})


def interval_text(value):
    return "n/a" if value is None else f"[{bench.number(value[0])}, {bench.number(value[1])}]"


def own_report(measurements, locale):
    cis, difference_cis = intervals(measurements)
    rows = []
    for result, model_cis in zip(measurements, cis, strict=True):
        point = statistics_for(result, range(len(result.samples)))
        metrics = []
        for value, ci in zip(point, model_cis, strict=True):
            metrics.append(f"{bench.number(value)} {interval_text(ci)}")
        rows.append((result.backend, len(result.samples), *metrics))
    labels = {bench.BACKENDS[0]: "Apple", bench.BACKENDS[1]: "Whisper", bench.PARAKEET_BACKEND: "Parakeet"}
    names = [labels.get(result.backend, result.backend) for result in measurements]
    comparisons = []
    for (candidate, baseline), ci in difference_cis.items():
        difference = (measurements[candidate].overall["metrics"]["wer"]["value"]
                      - measurements[baseline].overall["metrics"]["wer"]["value"])
        comparisons.append(
            f"\nPaired WER difference ({names[candidate]} minus {names[baseline]}): "
            f"{bench.number(difference)} {interval_text(ci)}.\n"
            f"Negative favors {names[candidate]}; positive favors {names[baseline]}. "
            "An interval including zero does not resolve a difference.\n")
    has_parakeet = any(result.backend == bench.PARAKEET_BACKEND for result in measurements)
    parakeet_method = ("Parakeet uses LocalParakeet's load/decode recipe on float arrays from the same untrimmed WAVs; "
                       "Parakeet detects language without a locale hint. " if has_parakeet else "")
    not_measured = "Qwen3-ASR and Nemotron" if has_parakeet else "Qwen3-ASR, Nemotron and Parakeet"
    audit_rows = []
    for result in measurements:
        if result.numeric_form_audit is not None:
            matched, eligible = result.numeric_form_audit
            audit_rows.append((result.backend, matched if eligible else "n/a", eligible))
    audit = ("\n## Numeric-form audit\n\n"
             + bench.table(("backend", "matched_clips", "eligible_clips"), audit_rows)
             + "\nLiteral ordered digit-written expression lists must match exactly, including extras and repeats. "
             "Only clips with reference expressions are eligible; no eligible clips means n/a. "
             "This is not semantic equivalence for spoken number words. "
             "Counts only; WER/CER and required-term metrics are unchanged.\n" if audit_rows else "")
    return (
        "# Own-voice benchmark\n\n"
        + bench.table(("backend", "n", "wer [95% CI]", "cer [95% CI]",
                       "required_term_hit_rate [95% CI]", "median_latency_s [95% CI]"), rows)
        + "".join(comparisons) + "\n"
        + f"Locale: {locale}; Whisper language: {locale.split('-')[0]}. "
        + f"Whisper revision: `{bench.WHISPER_REVISION}`.\n"
        + (f"Parakeet revision: `{bench.PARAKEET_REVISION}`.\n" if has_parakeet else "")
        + "Date and macOS version are in the companion platform report from this same run.\n\n"
        + f"Method: identical files and manifest order; {names[0]} first, then {', then '.join(names[1:])}; "
        + "one pass, no warmup. "
        + "Latency is file-to-final-text wall time, including the first decode's cold load, excluding model setup, "
        + "authorization, scoring and capture. It is not release-to-paste latency. "
        + "No glossary, contextual terms, dictionary, VAD trimming, cleanup or personal settings are applied. "
        + parakeet_method
        + "This measures the default Whisper model through benchmark._transcribe, not the complete Scrivo app.\n\n"
        + "WER/CER are micro averages from evaluation.py; required-term hit rate is matched/required terms "
        + "(n/a when none were annotated). Case/punctuation normalization follows that module. "
        + f"Intervals use {BOOTSTRAP_REPLICATES} paired clip bootstrap draws, seed {BOOTSTRAP_SEED}, "
        + "and the 2.5/97.5 percentiles. Clips, not words, are the resampling unit. "
        + "A term interval is n/a if any resample lacks annotated terms.\n\n"
        + "Limits: intended for one speaker and one microphone, as recorded by the person supplying the clips; "
        + "these attributes are not verified by the runner. A small, self-selected corpus and correlated clips "
        + "limit generalization; bootstrap intervals do not represent between-speaker uncertainty. "
        + "Backend order, thermal state and cold caches can affect latency. "
        + "This is Apple's Speech framework (on-device), not a macOS Dictation product test. "
        + f"{not_measured} were not benchmarked. "
        + "No failure examples or per-clip outputs are retained or quoted.\n"
        + audit
    )


def main(argv=None):
    parser = bench.parser("Produce own-voice and platform aggregate reports from the same recognition pass.")
    parser.add_argument("--out", required=True, metavar="OWN_REPORT")
    parser.add_argument("--platform-out", required=True, metavar="PLATFORM_REPORT")
    parser.add_argument("--numeric-form-audit", action="store_true",
                        help="Add aggregate literal numeric-expression matched/eligible clip counts")
    args = parser.parse_args(argv)
    try:
        targets = bench.check_outputs([args.out, args.platform_out])
        with bench.quiet_backends():
            rows = bench.load_rows(args.manifest, args.locale)
        if args.dry_run:
            print(f"Validated {len(rows)} clips. No recognition was run; no reports were written.")
            return 0
        ensure_host()
        cache = args.cache_dir or Path(args.manifest).expanduser().resolve().parent / "models"
        measured = bench.measure(rows, locale=args.locale, cache_dir=cache, with_parakeet=args.with_parakeet,
                                 numeric_form_audit=args.numeric_form_audit)
        platform = bench.platform_report(measured)
        own = own_report(measured, args.locale)
        bench.write_reports([(targets[0], own), (targets[1], platform)])
        print(platform, end="")
        print("Wrote both aggregate reports from the same recognition pass.")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        return bench.report_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
