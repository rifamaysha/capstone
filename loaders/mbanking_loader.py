"""Parser dataset M-Banking (screenshot, primer, anonim)."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .schema import UnifiedRecord

logger = logging.getLogger(__name__)


def _parse_amount(s: str | None) -> float | None:
    """Parse '200000', '200,000', 'Rp 200.000' → 200000.0"""
    if not s:
        return None
    cleaned = re.sub(r"[^\d-]", "", str(s))
    if not cleaned or cleaned == "-":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def load_mbanking(jsonl_path: Path, image_dir: Path) -> list[UnifiedRecord]:
    """Load M-Banking dataset menjadi UnifiedRecord.

    Recipient-nya bisa berisi merchant + kategori informal
    (mis. 'FAMILY DENTAL CARE BUAH B - BANDUNG'), kita pakai langsung
    sebagai text_for_classification untuk training IndoBERT nanti.

    Args:
        jsonl_path: Path ke donut_train.jsonl mbanking.
        image_dir: Folder gambar processed (data_processed/mbanking/).
            Bisa berisi subfolder per bank (bca/, mandiri/, dll.).

    Returns:
        List of UnifiedRecord.
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")

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
                gt = json.loads(row["ground_truth"])
                # Format bisa "{gt_parse: {...}}" atau langsung "{...}"
                gt_parse = gt.get("gt_parse", gt)

                # File mungkin nested di subfolder bank → cari rekursif
                img_stem = Path(file_name).stem
                matches = list(image_dir.rglob(f"{img_stem}.png"))
                if not matches:
                    skipped += 1
                    continue
                img_path = matches[0]

                # Bank dari nama folder parent (kalau nested di mbanking/bca/...)
                bank = (img_path.parent.name
                        if img_path.parent.resolve() != image_dir.resolve()
                        else None)

                recipient = (gt_parse.get("recipient") or "").strip()

                records.append(UnifiedRecord(
                    image_path=img_path,
                    source="mbanking",
                    language="id",
                    currency="IDR",
                    merchant=recipient,
                    transaction_date=gt_parse.get("date"),
                    total_amount=_parse_amount(gt_parse.get("amount")),
                    text_for_classification=recipient,
                    raw_metadata={"bank": bank, "raw_recipient": recipient},
                ))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning("Skip line %d in %s: %s",
                               line_no, jsonl_path.name, exc)
                skipped += 1

    logger.info("M-Banking: loaded %d records (skipped %d)", len(records), skipped)
    return records