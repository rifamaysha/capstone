"""Parser dataset Kaggle (struk Inggris US/CA)."""
from __future__ import annotations

import ast
import csv
import json
import logging
from pathlib import Path

from .schema import UnifiedRecord

logger = logging.getLogger(__name__)


def _parse_jsonl_gt(gt_str: str) -> dict:
    """Parse field 'ground_truth' dari donut_train.jsonl (escaped JSON)."""
    parsed = json.loads(gt_str)
    # Ada wrapper "gt_parse" pada beberapa baris; un-wrap kalau ada.
    return parsed.get("gt_parse", parsed)


def _parse_csv_gt(gt_str: str) -> dict:
    """Parse field 'gt_json' dari train_list.csv / test_list.csv.

    Format = Python dict string (single quote), bukan JSON valid.
    Pakai ast.literal_eval (aman, hanya mengeksekusi literal).
    """
    return ast.literal_eval(gt_str)


def _annotations_to_text(annotations: list[dict]) -> str:
    """Concat semua text dari list of annotations → 1 string untuk classifier."""
    return " ".join(a.get("text", "") for a in annotations if a.get("text"))


def load_kaggle(
    jsonl_path: Path,
    image_dir: Path,
    csv_path: Path | None = None,
) -> list[UnifiedRecord]:
    """Load Kaggle dataset menjadi UnifiedRecord.

    Args:
        jsonl_path: Path ke donut_train.jsonl.
        image_dir: Folder gambar processed (data_processed/kaggle/).
        csv_path: Opsional, train_list.csv/test_list.csv untuk lookup kategori
            yang lebih reliable (kolom 'category' di CSV cleaner daripada metadata).

    Returns:
        List of UnifiedRecord. File yang gambar processed-nya tidak ada di-skip.

    Raises:
        FileNotFoundError: Bila jsonl atau image_dir tidak ada.
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")

    # Optional: bangun lookup kategori dari CSV (lebih bersih dari metadata.category)
    csv_lookup: dict[str, str] = {}
    if csv_path and csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                csv_lookup[row["file_name"]] = (row.get("category") or "").strip()
        logger.info("Loaded %d categories from %s", len(csv_lookup), csv_path.name)

    records: list[UnifiedRecord] = []
    skipped = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                file_name = row["file_name"]
                gt = _parse_jsonl_gt(row["ground_truth"])

                metadata = gt.get("metadata", {})
                annotations = gt.get("annotations", [])

                # Preprocessing menyamakan semua extension → .png
                img_path = image_dir / f"{Path(file_name).stem}.png"
                if not img_path.exists():
                    skipped += 1
                    continue

                category = csv_lookup.get(file_name) or metadata.get("category")

                records.append(UnifiedRecord(
                    image_path=img_path,
                    source="kaggle",
                    language="en",
                    category=category,
                    currency="USD",       # ASUMSI: mayoritas US; CA-spesifik di-tag belakangan
                    transaction_date=metadata.get("year"),  # Kaggle hanya kasih year
                    text_for_classification=_annotations_to_text(annotations),
                    raw_metadata={
                        "metadata": metadata,
                        "n_annotations": len(annotations),
                    },
                ))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning("Skip line %d in %s: %s",
                               line_no, jsonl_path.name, exc)
                skipped += 1

    logger.info("Kaggle: loaded %d records (skipped %d)", len(records), skipped)
    return records