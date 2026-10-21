"""Build a document-issuer split 10-K extraction set from vendored passages."""

from __future__ import annotations

import ast
import hashlib
import json
import random
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from phase4.constants import (
    CAPITALCITE_COMMIT,
    CAPITALCITE_REPO,
    CEILING,
    CHUNK_RADIUS,
    ISSUER_ALIASES,
    MAX_CHUNK_CHARS,
    MIN_DISTINCT_NUMS,
    MIN_EVAL,
    MIN_TRAIN,
    MODEL_ID,
    MODEL_REVISION,
    NEAR_DUP_THRESHOLD,
    SHINGLE_N,
    SPLIT_SEED,
    TASK_ID,
)
from phase4.grade import BARE_FIGURE, canonicalize, prompt_template_static, render_prompt
from phase4.paths import BuildStopped, refuse_phase3

NUM = re.compile(
    r"\(\s*\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*\)|"
    r"\(\s*\$?\d+\.\d+\s*\)|"
    r"\(\s*\$?\d{2,}\s*\)|"
    r"\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?|"
    r"\$?\d+\.\d+%?|"
    r"\d+%|"
    r"\$?\d{2,}"
)
YEAR = re.compile(r"\b(20[0-2]\d|19[89]\d)\b")
YEAR_LINE = re.compile(r"^(20[0-2]\d|19[89]\d)$")
CELL_NUM = re.compile(
    r"\(\s*\$?\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*\)|"
    r"\(\s*\$?\s*\d+\.\d+\s*\)|"
    r"\$\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?|"
    r"\$\s*\d+(?:\.\d+)?|"
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|"
    r"\d+\.\d+|"
    r"\d{2,}"
)
LABEL_LINE = re.compile(r"^[A-Za-z][A-Za-z0-9 &'/().,%-]{1,80}$")
SKIP_LABELS = {
    "nm",
    "change",
    "assets",
    "liabilities",
    "liabilities and equity",
    "page",
    "consolidated",
}
ACCESSION = re.compile(r"^(\d{10})")
WRITTEN_NAMES = (
    "train.jsonl",
    "eval.jsonl",
    "manifest.json",
    "contamination.json",
    "grade.py",
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vendor_root(repo_root: Path) -> Path:
    return repo_root / "evals" / "phase4" / "capitalcite"


def refuse_if_chunks_jsonl(source: Path) -> None:
    roots = [source, *list(source.parents)[:5]]
    for root in roots:
        parsed = root / "data" / "parsed"
        if not parsed.is_dir():
            continue
        hits = sorted(parsed.rglob("chunks.jsonl"))
        if hits:
            raise BuildStopped(
                f"chunks.jsonl present at {hits[0]}; refuse to hide a second label source"
            )


def assert_live_capitalcite_pin(source: Path) -> None:
    for root in [source, *list(source.parents)[:5]]:
        passages = root / "evals" / "golden" / "passages"
        git_dir = root / ".git"
        if passages.is_dir() and git_dir.exists():
            sha = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                text=True,
            ).strip()
            if sha != CAPITALCITE_COMMIT:
                raise BuildStopped(
                    f"capitalcite HEAD {sha} is not the pinned {CAPITALCITE_COMMIT}"
                )
            return


def split_hash(eval_items: list[dict[str, str]], train_items: list[dict[str, str]]) -> str:
    held = "\n".join(sorted(item["item_id"] for item in eval_items))
    train = "\n".join(sorted(item["item_id"] for item in train_items))
    return sha256_text(held + "\n---\n" + train)


def load_passages(source: Path) -> dict[str, str]:
    directory = source / "passages"
    if not directory.is_dir():
        raise BuildStopped(f"passages directory missing: {directory}")
    return {path.stem: path.read_text(encoding="utf-8") for path in sorted(directory.glob("*.txt"))}


def normalize_doc(doc_id: str) -> str:
    return doc_id.replace("/", "__")


def issuer_of(doc_id: str) -> str:
    raw = normalize_doc(doc_id)
    accession = ACCESSION.match(raw)
    if accession:
        key = accession.group(1)
        return ISSUER_ALIASES.get(key, key)
    token = re.split(r"[_\W]", raw, maxsplit=1)[0].upper()
    return ISSUER_ALIASES.get(token, token)


def compact_text(text: str) -> str:
    text = re.sub(r"\(\s*\$?([\d,.]+)\s*\)", r"-\1", text)
    return re.sub(r"[\s$,]", "", text)


def distinct_nums(text: str) -> int:
    return len({canonicalize(match) for match in NUM.findall(text) if canonicalize(match)})


def window_around(text: str, needle: str, radius: int = CHUNK_RADIUS) -> str:
    if not needle:
        return text[:MAX_CHUNK_CHARS]
    index = text.find(needle)
    if index < 0:
        index = text.lower().find(needle.lower())
    if index < 0:
        return text[:MAX_CHUNK_CHARS]
    start = max(0, index - radius)
    end = min(len(text), index + len(needle) + radius)
    chunk = text[start:end]
    if len(chunk) > MAX_CHUNK_CHARS:
        extra = len(chunk) - MAX_CHUNK_CHARS
        chunk = chunk[extra // 2 : extra // 2 + MAX_CHUNK_CHARS]
    return chunk


def _passage_for(doc_id: str, passages: dict[str, str]) -> str | None:
    if doc_id in passages:
        return passages[doc_id]
    alt = normalize_doc(doc_id)
    if alt in passages:
        return passages[alt]
    key = alt.replace(".pdf", "")
    for name, text in passages.items():
        if name.replace(".pdf", "") == key or name.startswith(key[:18]):
            return text
    return None


def _item(
    *,
    item_id: str,
    doc_id: str,
    question: str,
    value: str,
    chunk: str,
    source: str,
) -> dict[str, str] | None:
    completion = canonicalize(value)
    if not completion or not BARE_FIGURE.fullmatch(completion):
        return None
    if distinct_nums(chunk) < MIN_DISTINCT_NUMS:
        return None
    if completion not in compact_text(chunk):
        return None
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", item_id).strip("-")[:96]
    return {
        "item_id": slug,
        "doc_id": normalize_doc(doc_id),
        "issuer": issuer_of(doc_id),
        "question": question.strip(),
        "completion": completion,
        "chunk": chunk,
        "source": source,
    }


def golden_items(source: Path, passages: dict[str, str]) -> list[dict[str, str]]:
    path = source / "v1.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("type") not in {"numeric", "extractive"}:
            continue
        evidence = row.get("evidence") or []
        if not evidence:
            continue
        doc_id = evidence[0]["doc_id"]
        text = _passage_for(doc_id, passages)
        if text is None:
            continue
        answer = str(row["answer"])
        if canonicalize(answer) not in canonicalize(text) and answer not in text:
            continue
        needle = evidence[0].get("quote") or answer
        chunk = window_around(text, needle if needle in text else answer)
        item = _item(
            item_id=f"golden-{row['id']}",
            doc_id=doc_id,
            question=str(row["question"]),
            value=answer,
            chunk=chunk,
            source="golden",
        )
        if item:
            rows.append(item)
    return rows


def table_items(passages: dict[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, text in passages.items():
        for match in re.finditer(r"\[\[.*?\]\]", text, re.S):
            try:
                grid = ast.literal_eval(match.group(0))
            except (ValueError, SyntaxError):
                continue
            if not (isinstance(grid, list) and grid and isinstance(grid[0], list)):
                continue
            header = [str(cell) for cell in grid[0]]
            years = [(YEAR.findall(cell) or [None])[0] for cell in header]
            table_text = window_around(text, match.group(0)[:80])
            for row in grid[1:]:
                if not row:
                    continue
                label = str(row[0]).strip()
                if not label:
                    continue
                for index, cell in enumerate(row[1:], start=1):
                    if index >= len(years) or years[index] is None:
                        continue
                    found = CELL_NUM.findall(str(cell).strip())
                    if not found:
                        found = NUM.findall(str(cell).replace(" ", ""))
                    if len(found) != 1:
                        continue
                    year = years[index]
                    item = _item(
                        item_id=f"table-{name}-{index}-{label}-{year}",
                        doc_id=name,
                        question=f"What was the {label} value at year-end {year}?",
                        value=found[0],
                        chunk=table_text,
                        source="json_table",
                    )
                    if item:
                        rows.append(item)
    return rows


def _collect_numbers(lines: list[str], start: int) -> tuple[list[str], int]:
    nums: list[str] = []
    index = start
    while index < len(lines):
        line = lines[index]
        if YEAR_LINE.match(line):
            break
        if LABEL_LINE.match(line) and not NUM.search(line.replace(",", "")):
            break
        found = NUM.findall(line.replace(" ", ""))
        if found:
            nums.extend(found)
        index += 1
    return nums, index


def _named_columns(lines: list[str], year_at: int) -> list[str]:
    headers: list[str] = []
    index = year_at - 1
    while index >= 0:
        line = lines[index]
        if YEAR_LINE.match(line):
            break
        if NUM.search(line.replace(",", "")):
            break
        if not LABEL_LINE.match(line):
            break
        headers.append(line)
        index -= 1
    headers.reverse()
    if len(headers) >= 2:
        return headers
    return []


def _year_run(lines: list[str], start: int) -> tuple[list[str], int]:
    years: list[str] = []
    index = start
    while index < len(lines) and YEAR_LINE.match(lines[index]):
        year = lines[index]
        if year in years:
            break
        years.append(year)
        index += 1
    return years, index


def _skip_label(label: str) -> bool:
    return label.lower().strip() in SKIP_LABELS or YEAR_LINE.match(label) is not None


def year_column_items(passages: dict[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, text in passages.items():
        if distinct_nums(text) < MIN_DISTINCT_NUMS:
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        index = 0
        years: list[str] = []
        named: list[str] = []
        while index < len(lines):
            line = lines[index]
            if YEAR_LINE.match(line):
                run, next_index = _year_run(lines, index)
                if len(run) >= 2:
                    years = run
                    named = []
                    index = next_index
                    continue
                years = [line]
                found_named = _named_columns(lines, index)
                if found_named:
                    named = found_named
                index += 1
                continue
            if not years or not LABEL_LINE.match(line) or NUM.search(line.replace(",", "")):
                index += 1
                continue
            label = line
            nums, next_index = _collect_numbers(lines, index + 1)
            chunk = window_around(text, label)
            if _skip_label(label):
                index = max(next_index, index + 1)
                continue
            if named and len(nums) == len(named) and len(years) == 1:
                year = years[0]
                for column, raw in zip(named, nums):
                    item = _item(
                        item_id=f"col-{name}-{label}-{column}-{year}",
                        doc_id=name,
                        question=f"What was {label} for {column} at year-end {year}?",
                        value=raw,
                        chunk=chunk,
                        source="named_column",
                    )
                    if item:
                        rows.append(item)
            elif not named and len(nums) == len(years) and 2 <= len(years) <= 4:
                for year, raw in zip(years, nums):
                    item = _item(
                        item_id=f"row-{name}-{label}-{year}",
                        doc_id=name,
                        question=f"What was {label} at year-end {year}?",
                        value=raw,
                        chunk=chunk,
                        source="year_column",
                    )
                    if item:
                        rows.append(item)
            index = max(next_index, index + 1)
    return rows


def dedupe(items: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, str]] = []
    for item in items:
        key = (item["doc_id"], item["question"], item["completion"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def split_by_issuer(
    items: list[dict[str, str]], seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str], list[str]]:
    by_issuer: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in items:
        by_issuer[item["issuer"]].append(item)
    issuers = sorted(by_issuer)
    rng = random.Random(seed)
    rng.shuffle(issuers)
    eval_issuers: list[str] = []
    train_issuers: list[str] = []
    eval_n = 0
    for issuer in issuers:
        if eval_n < MIN_EVAL:
            eval_issuers.append(issuer)
            eval_n += len(by_issuer[issuer])
        else:
            train_issuers.append(issuer)
    eval_items = sorted(
        [item for issuer in eval_issuers for item in by_issuer[issuer]],
        key=lambda item: item["item_id"],
    )
    train_items = sorted(
        [item for issuer in train_issuers for item in by_issuer[issuer]],
        key=lambda item: item["item_id"],
    )
    return train_items, eval_items, train_issuers, eval_issuers


def shingles(text: str, n: int = SHINGLE_N) -> set[str]:
    norm = re.sub(r"\s+", " ", text.lower()).strip()
    if len(norm) < n:
        return {norm} if norm else set()
    return {norm[i : i + n] for i in range(len(norm) - n + 1)}


def jaccard(left: str, right: str) -> float:
    a = shingles(left)
    b = shingles(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def contamination_report(train: list[dict[str, str]], eval_items: list[dict[str, str]]) -> dict:
    train_chunks = {item["chunk"] for item in train}
    eval_chunks = {item["chunk"] for item in eval_items}
    exact = sorted(train_chunks & eval_chunks)
    near: list[dict[str, str | float]] = []
    train_unique = list(train_chunks)
    for ev in eval_chunks:
        for tr in train_unique:
            score = jaccard(ev, tr)
            if score >= NEAR_DUP_THRESHOLD:
                near.append(
                    {
                        "eval_sha256": sha256_text(ev),
                        "train_sha256": sha256_text(tr),
                        "jaccard": round(score, 4),
                    }
                )
    template = prompt_template_static()
    template_canon = canonicalize(template)
    leaked: list[str] = []
    for item in eval_items:
        if item["completion"] and item["completion"] in template_canon:
            leaked.append(item["item_id"])
        prompt = render_prompt(item["chunk"], item["question"])
        if item["chunk"] and item["chunk"] in template:
            leaked.append(item["item_id"])
        del prompt
    leaked = sorted(set(leaked))
    ok = not exact and not near and not leaked
    return {
        "exact_chunk_overlap": len(exact),
        "exact_chunk_shas": [sha256_text(chunk) for chunk in exact],
        "near_duplicate_pairs": near,
        "near_duplicate_threshold": NEAR_DUP_THRESHOLD,
        "eval_in_prompt_template": leaked,
        "ok": ok,
    }


def _jsonl(items: list[dict[str, str]], split: str) -> str:
    lines = []
    for item in items:
        row = dict(item)
        row["split"] = split
        row["task_id"] = TASK_ID
        row["prompt"] = render_prompt(item["chunk"], item["question"])
        lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return "\n".join(lines) + ("\n" if lines else "")


def collect_items(source: Path) -> list[dict[str, str]]:
    refuse_if_chunks_jsonl(source)
    assert_live_capitalcite_pin(source)
    passages = load_passages(source)
    if not passages:
        raise BuildStopped(f"no passages under {source}")
    items = dedupe(golden_items(source, passages) + table_items(passages) + year_column_items(passages))
    if not items:
        raise BuildStopped("builder produced no items")
    return items


def materialize(source: Path, seed: int = SPLIT_SEED) -> dict:
    items = collect_items(source)
    train, ev, train_issuers, eval_issuers = split_by_issuer(items, seed)
    if len(ev) < MIN_EVAL:
        raise BuildStopped(f"eval n {len(ev)} is below {MIN_EVAL}")
    if len(train) < MIN_TRAIN:
        raise BuildStopped(f"train n {len(train)} is below {MIN_TRAIN}")
    report = contamination_report(train, ev)
    if not report["ok"]:
        raise BuildStopped(f"contamination check failed: {json.dumps(report, sort_keys=True)}")
    return {
        "train": train,
        "eval": ev,
        "train_issuers": train_issuers,
        "eval_issuers": eval_issuers,
        "contamination": report,
        "seed": seed,
        "source": str(source),
        "n_items": len(items),
    }


def write_dataset(payload: dict, dest: Path, source: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    train_text = _jsonl(payload["train"], "train")
    eval_text = _jsonl(payload["eval"], "eval")
    (dest / "train.jsonl").write_text(train_text, encoding="utf-8")
    (dest / "eval.jsonl").write_text(eval_text, encoding="utf-8")
    grade_src = Path(__file__).resolve().parent / "grade.py"
    shutil.copyfile(grade_src, dest / "grade.py")
    manifest = {
        "task_id": TASK_ID,
        "capitalcite_repo": CAPITALCITE_REPO,
        "capitalcite_commit": CAPITALCITE_COMMIT,
        "source_kind": "evals/golden/passages",
        "chunks_jsonl": "absent",
        "source_dir": "evals/phase4/capitalcite",
        "split_seed": payload["seed"],
        "n_items": payload["n_items"],
        "n_train": len(payload["train"]),
        "n_eval": len(payload["eval"]),
        "train_issuers": payload["train_issuers"],
        "eval_issuers": payload["eval_issuers"],
        "train_docs": sorted({item["doc_id"] for item in payload["train"]}),
        "eval_docs": sorted({item["doc_id"] for item in payload["eval"]}),
        "train_sha256": sha256_text(train_text),
        "eval_sha256": sha256_text(eval_text),
        "grade_sha256": sha256_file(dest / "grade.py"),
        "split_issuers_sha256": sha256_text("\n".join(sorted(payload["eval_issuers"]))),
        "split_hash": split_hash(payload["eval"], payload["train"]),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "ceiling": CEILING,
        "builder": "phase4.build",
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (dest / "contamination.json").write_text(
        json.dumps(payload["contamination"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build(source: Path, dest: Path, seed: int = SPLIT_SEED) -> dict:
    refuse_phase3(dest)
    dest = Path(dest)
    source = Path(source)
    payload = materialize(source, seed)
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{dest.name}.phase4-tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        manifest = write_dataset(payload, tmp, source)
        for name in WRITTEN_NAMES:
            target = dest / name
            source_file = tmp / name
            if source_file.is_file():
                shutil.copyfile(source_file, target)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp)
        raise
    shutil.rmtree(tmp)
    return manifest


def build_default(repo_root: Path, dest: Path | None = None, seed: int = SPLIT_SEED) -> dict:
    source = vendor_root(repo_root)
    if dest is None:
        dest = repo_root / "kaggle" / "phase4_data"
    return build(source, dest, seed)
