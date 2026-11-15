"""Compare on-device Apple Speech with the default pinned MLX Whisper model."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import bench_common as bench  # noqa: E402
from scripts.bench_host import ensure_host  # noqa: E402


def main(argv=None):
    parser = bench.parser("Write an aggregate-only Apple Speech / MLX Whisper comparison.")
    parser.add_argument("--out", required=True, metavar="REPORT")
    args = parser.parse_args(argv)
    try:
        targets = bench.check_outputs([args.out])
        with bench.quiet_backends():
            rows = bench.load_rows(args.manifest, args.locale)
        if args.dry_run:
            print(f"Validated {len(rows)} clips. No recognition was run; no report was written.")
            return 0
        cache = args.cache_dir or Path(args.manifest).expanduser().resolve().parent / "models"
        ensure_host()
        results = bench.measure(rows, locale=args.locale, cache_dir=cache)
        report = bench.platform_report(results)
        bench.write_reports([(targets[0], report)])
        print(report, end="")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        return bench.report_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
