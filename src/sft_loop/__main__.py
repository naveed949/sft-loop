"""Print one fail-closed unsupported report. Does not train."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sft_loop.report import unsupported_report, validate_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Emit a schema-valid unsupported report. "
            "Does not train and does not write before/after metrics."
        )
    )
    parser.add_argument("--run-id", default="scaffold-unsupported")
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional path for the JSON report. Default is stdout.",
    )
    args = parser.parse_args(argv)
    try:
        report = unsupported_report(run_id=args.run_id)
        validate_report(report)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    payload = json.dumps(report, indent=2) + "\n"
    if args.out is None:
        sys.stdout.write(payload)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
