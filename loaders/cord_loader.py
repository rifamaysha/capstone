"""Parser dataset CORD v2 (struk Indonesia)."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from .schema import TransactionItem, UnifiedRecord

logger = logging.getLogger(__name__)


def _to_str(value: Any) -> str | None:
    """Coerce nilai field CORD (str / list / None / int) ke string.

    CORD inkonsisten: field 'nm', 'price', 'cnt' bisa berupa string
    ('Nasi Goreng') ATAU list (['Nasi', 'Goreng']) — yang terakhir terjadi
    ketika OCR mendeteksi multi-line untuk satu menu. Kita gabungkan list
    jadi 1 string yang dipisah spasi.
    """
    if value is None:
        return None
    if isinstance(value, list):
        joined = " ".join(str(x) for x in value if x not in (None, ""))
        return joined or None
    return str(value)


def _parse_idr(value: Any) -> float | None:
    """Parse harga Rupiah ('75,000', '1.591.600', list, dll.) → float."""
    s = _to_str(value)
    if not s:
        return None
    cleaned = re.sub(r"[^\d-]", "", s)
    if not cleaned or cleaned == "-":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_count(value: Any) -> float | None:
    """Parse field cnt CORD ('1 x', '2', '0.5 kg') → quantity float."""
    s = _to_str(value)
    if not s:
        return None
    m = re.search(r"\d+(?:[.,]\d+)?", s)
    return float(m.group().replace(",", ".")) if m else None


def load_cord(jsonl_path: Path, image_dir: Path) -> list[UnifiedRecord]:
    """Load CORD v2 dataset menjadi UnifiedRecord.

    Args:
        jsonl_path: Path ke donut_train.jsonl CORD.
        image_dir: Folder gambar processed (data_processed/huggingface/).

    Returns:
        List of UnifiedRecord, lengkap dengan items[] tiap struk.
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
                img_rel = row.get("image_path") or row.get("file_name")
                if not img_rel:
                    skipped += 1
                    continue

                gt = json.loads(row["ground_truth"])

                # Cari gambar processed; rglob karena CORD bisa nested
                img_stem = Path(img_rel).stem
                matches = list(image_dir.rglob(f"{img_stem}.png"))
                if not matches:
                    skipped += 1
                    continue
                img_path = matches[0]

                # Parse menu — bisa list, dict tunggal, atau hilang
                menu = gt.get("menu", [])
                if isinstance(menu, dict):
                    menu = [menu]

                items: list[TransactionItem] = []
                names: list[str] = []
                for m in menu:
                    if not isinstance(m, dict):
                        continue
                    name = (_to_str(m.get("nm")) or "").strip()
                    if not name:
                        continue
                    items.append(TransactionItem(
                        name=name,
                        quantity=_parse_count(m.get("cnt")),
                        unit_price=_parse_idr(m.get("unitprice")),
                        total_price=_parse_idr(m.get("price")),
                    ))
                    names.append(name)

                # Total — sub_total/total bisa juga list/dict/string
                total_data = gt.get("total", {})
                total_price_raw = (
                    total_data.get("total_price")
                    if isinstance(total_data, dict)
                    else None
                )
                total_amount = _parse_idr(total_price_raw)

                records.append(UnifiedRecord(
                    image_path=img_path,
                    source="cord",
                    language="id",
                    currency="IDR",
                    total_amount=total_amount,
                    items=items,
                    text_for_classification=" ".join(names),
                    raw_metadata={"sub_total": gt.get("sub_total", {})},
                ))
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                logger.warning("Skip line %d in %s: %s",
                               line_no, jsonl_path.name, exc)
                skipped += 1

    logger.info("CORD: loaded %d records (skipped %d)", len(records), skipped)
    return records