"""CLI for the Phase 4 sibling: build, fence, assemble, probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from phase4.build import build_default
from phase4.constants import SPLIT_SEED
from phase4.fence import fence_violations, iter_fenced_files, product_imports_phase4
from phase4.paths import BuildStopped, Phase3PathRefused, Phase4Unsupported, RefusedCeiling
from phase4.run import local_scaler_probe


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "schema" / "run_report.schema.json").is_file():
            return parent
    raise SystemExit("could not find repo root")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4 sibling tools. Does not drive sft_loop.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build_p = sub.add_parser("build")
    build_p.add_argument("--dest", type=Path, default=None)
    build_p.add_argument("--seed", type=int, default=SPLIT_SEED)
    fence_p = sub.add_parser("fence")
    probe_p = sub.add_parser("probe")
    probe_p.add_argument("--out", type=Path, required=True)
    assemble_p = sub.add_parser("assemble")
    assemble_p.add_argument("--before", type=Path, required=True)
    assemble_p.add_argument("--after", type=Path, required=True)
    assemble_p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    root = _repo_root()
    try:
        if args.cmd == "build":
            manifest = build_default(root, dest=args.dest, seed=args.seed)
            sys.stdout.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            return 0
        if args.cmd == "fence":
            banned = []
            for path in iter_fenced_files(root):
                hits = fence_violations(path)
                if hits:
                    banned.append({"path": path.relative_to(root).as_posix(), "imports": hits})
            leaked = product_imports_phase4(root)
            sys.stdout.write(json.dumps({"banned_imports": banned, "product_imports_phase4": leaked}, indent=2) + "\n")
            return 1 if banned or leaked else 0
        if args.cmd == "probe":
            body = local_scaler_probe(args.out)
            sys.stdout.write(json.dumps(body, indent=2, sort_keys=True) + "\n")
            return 0
        if args.cmd == "assemble":
            from phase4.assemble import assemble

            report = assemble(args.before, args.after, args.out)
            sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
            return 0
    except (BuildStopped, Phase3PathRefused, Phase4Unsupported, RefusedCeiling) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
