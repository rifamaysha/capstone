"""Apply auto-labeler ke train/val/test JSONL → tulis versi labeled.

Output:
- data_processed/unified/train_labeled.jsonl
- data_processed/unified/val_labeled.jsonl
- data_processed/unified/test_labeled.jsonl

Setiap baris ditambahi field `category_label` (salah satu dari 8 kategori).
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from indobert import CATEGORIES, CATEGORY_DISPLAY, label_record

logger = logging.getLogger(__name__)


def label_split(in_path: Path, out_path: Path) -> Counter:
    """Baca JSONL, label tiap record, tulis hasil. Return counter label."""
    counter: Counter = Counter()
    with open(in_path, encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                label = label_record(record)
                record["category_label"] = label
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                counter[label] += 1
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Skip line %d in %s: %s",
                               line_no, in_path.name, exc)
    return counter


def print_distribution(name: str, counter: Counter, total: int) -> None:
    """Print breakdown kategori dalam tabel."""
    print(f"\n[{name}] Total: {total}")
    print(f"  {'Category':25s} {'Count':>6s}  Pct")
    print(f"  {'-'*25} {'-'*6}  ----")
    for cat in CATEGORIES:
        n = counter.get(cat, 0)
        pct = 100.0 * n / total if total else 0.0
        display = CATEGORY_DISPLAY[cat]
        print(f"  {display:25s} {n:6d}  {pct:5.1f}%")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    PROJECT_ROOT = Path(r"C:\Users\USER\Documents\CAPSTONE")
    UNIFIED = PROJECT_ROOT / "data_processed" / "unified"

    grand = Counter()
    grand_total = 0
    for split in ("train", "val", "test"):
        in_path  = UNIFIED / f"{split}.jsonl"
        out_path = UNIFIED / f"{split}_labeled.jsonl"
        if not in_path.exists():
            logger.error("Missing input: %s", in_path)
            continue
        counter = label_split(in_path, out_path)
        total = sum(counter.values())
        print_distribution(split, counter, total)
        grand.update(counter)
        grand_total += total
        logger.info("Wrote %s (%d records)", out_path.name, total)

    print("\n" + "=" * 50)
    print_distribution("OVERALL", grand, grand_total)


if __name__ == "__main__":
    main()