"""AST fence: Phase 4 code never imports sft_loop or unsloth."""

from __future__ import annotations

import ast
from pathlib import Path

BANNED = ("sft_loop", "unsloth")


def _module_root(name: str) -> str:
    return name.split(".", 1)[0]


def imported_roots(source: str) -> set[str]:
    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(_module_root(alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(_module_root(node.module))
    return roots


def fence_violations(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    roots = imported_roots(source)
    return sorted(root for root in roots if root in BANNED)


def iter_fenced_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for rel in ("src/phase4", "kaggle/phase4_train", "kaggle/phase4_data"):
        directory = root / rel
        if not directory.exists():
            continue
        files.extend(sorted(directory.rglob("*.py")))
    return files


def product_imports_phase4(root: Path) -> list[str]:
    hits: list[str] = []
    src = root / "src" / "sft_loop"
    verify = root / ".cursor" / "skills" / "verify-sft-loop" / "scripts"
    for directory in (src, verify):
        if not directory.exists():
            continue
        for path in directory.rglob("*.py"):
            roots = imported_roots(path.read_text(encoding="utf-8"))
            if "phase4" in roots:
                hits.append(path.relative_to(root).as_posix())
    return hits
