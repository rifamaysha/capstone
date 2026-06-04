"""Build unified dataset: load 3 sumber → split stratified → JSONL siap DONUT."""
from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from loaders import (
    UnifiedRecord, encode_donut_target,
    load_cord, load_kaggle, load_mbanking,
)

logger = logging.getLogger(__name__)


def stratified_split(
    records: list[UnifiedRecord],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[list[UnifiedRecord], list[UnifiedRecord], list[UnifiedRecord]]:
    """Split stratified per source. Aman untuk source dengan jumlah kecil.

    Tidak pakai sklearn.train_test_split agar source berukuran <10
    tetap bisa di-split tanpa error.

    Args:
        records: Gabungan record dari semua sumber.
        ratios: (train, val, test) — harus jumlahnya 1.0.
        seed: Random seed untuk reproducibility.

    Returns:
        Tuple (train, val, test).

    Raises:
        ValueError: Bila ratios tidak jumlahnya 1.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios harus jumlahnya 1.0, got {sum(ratios)}")

    rng = random.Random(seed)
    by_source: dict[str, list[UnifiedRecord]] = defaultdict(list)
    for r in records:
        by_source[r.source].append(r)

    train: list[UnifiedRecord] = []
    val:   list[UnifiedRecord] = []
    test:  list[UnifiedRecord] = []

    for source, items in by_source.items():
        shuffled = items.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * ratios[0])
        n_val   = int(n * ratios[1])

        train.extend(shuffled[:n_train])
        val.extend  (shuffled[n_train : n_train + n_val])
        test.extend (shuffled[n_train + n_val :])

        logger.info(
            "[%s] %d → train=%d  val=%d  test=%d",
            source, n, n_train, n_val, n - n_train - n_val,
        )

    return train, val, test


def write_jsonl(
    records: Iterable[UnifiedRecord],
    path: Path,
    split_name: str,
) -> int:
    """Tulis records ke JSONL dengan field donut_target ter-encode.

    Setiap baris berisi semua field yang dibutuhkan oleh training pipeline:
    DONUT (image_path, donut_target) + IndoBERT (text_for_classification, category).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            entry = {
                "image_path": str(r.image_path),
                "donut_target": encode_donut_target(r),
                "source": r.source,
                "language": r.language,
                "category": r.category,
                "merchant": r.merchant,
                "transaction_date": r.transaction_date,
                "total_amount": r.total_amount,
                "currency": r.currency,
                "text_for_classification": r.text_for_classification,
                "n_items": len(r.items),
                "split": split_name,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    PROJECT_ROOT = Path(r"C:\Users\USER\Documents\CAPSTONE")
    PROCESSED   = PROJECT_ROOT / "data_processed"
    OUT_DIR     = PROCESSED / "unified"

    # ---------- Load 3 sumber ----------
    all_records: list[UnifiedRecord] = []
    all_records.extend(load_kaggle(
        jsonl_path = PROJECT_ROOT / "dataset" / "dataset_kaggle" / "donut_train.jsonl",
        image_dir  = PROCESSED    / "kaggle",
        csv_path   = PROJECT_ROOT / "dataset" / "dataset_kaggle" / "train_list.csv",
    ))
    all_records.extend(load_cord(
        jsonl_path = PROJECT_ROOT / "dataset" / "dataset_hf" / "donut_train.jsonl",
        image_dir  = PROCESSED    / "huggingface",
    ))
    all_records.extend(load_mbanking(
        jsonl_path = PROJECT_ROOT / "dataset" / "dataset_mbanking" / "donut_train.jsonl",
        image_dir  = PROCESSED    / "mbanking",
    ))

    # ---------- Ringkasan input ----------
    logger.info("=" * 72)
    logger.info("INPUT TOTAL: %d records", len(all_records))
    for src, n in Counter(r.source for r in all_records).items():
        logger.info("  %-10s: %d", src, n)

    # ---------- Split ----------
    logger.info("=" * 72)
    logger.info("SPLIT (stratified per source, ratio 80/10/10, seed=42)")
    train, val, test = stratified_split(all_records, (0.8, 0.1, 0.1), seed=42)

    # ---------- Tulis JSONL ----------
    logger.info("=" * 72)
    logger.info("WRITE → %s", OUT_DIR)
    n_train = write_jsonl(train, OUT_DIR / "train.jsonl", "train")
    n_val   = write_jsonl(val,   OUT_DIR / "val.jsonl",   "val")
    n_test  = write_jsonl(test,  OUT_DIR / "test.jsonl",  "test")
    logger.info("  train.jsonl: %d", n_train)
    logger.info("  val.jsonl  : %d", n_val)
    logger.info("  test.jsonl : %d", n_test)

    # ---------- Sample target per source ----------
    logger.info("=" * 72)
    logger.info("SAMPLE DONUT TARGETS")
    seen: set[str] = set()
    for r in train:
        if r.source in seen:
            continue
        seen.add(r.source)
        target = encode_donut_target(r)
        preview = target if len(target) <= 300 else target[:300] + "..."
        logger.info("[%s] %s", r.source, preview)
        if len(seen) == 3:
            break


if __name__ == "__main__":
    main()