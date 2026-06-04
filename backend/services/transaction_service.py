"""Transaction persistence service — reads/writes data/transactions.json."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ..schemas import TransactionCreate, TransactionOut

logger = logging.getLogger(__name__)

# Same file as Streamlit uses — shared source of truth
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "transactions.json"

CATEGORY_DISPLAY: dict[str, str] = {
    "makanan_minuman": "Makanan & Minuman",
    "transportasi": "Transportasi",
    "belanja": "Belanja & Retail",
    "hiburan": "Hiburan & Wisata",
    "kesehatan": "Kesehatan",
    "pendidikan": "Pendidikan",
    "tagihan": "Tagihan & Utilitas",
    "lainnya": "Lainnya",
}


def _load_raw() -> list[dict[str, Any]]:
    if not DATA_FILE.exists():
        return []
    try:
        with open(DATA_FILE, encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        logger.warning("transactions.json tidak bisa dibaca — mulai dari kosong")
        return []


def _save_raw(transactions: list[dict[str, Any]]) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(transactions, f, indent=2, ensure_ascii=False, default=str)


def _to_out(raw: dict[str, Any]) -> TransactionOut:
    cat = raw.get("category", "lainnya")
    merchant = raw.get("merchant", "")
    try:
        from .extraction_service import clean_merchant_candidate
        merchant = clean_merchant_candidate(merchant) or merchant
    except Exception:
        pass
    return TransactionOut(
        id=raw.get("id", 0),
        merchant=merchant,
        amount=float(raw.get("amount", 0)),
        date=raw.get("date", ""),
        category=cat,
        category_display=CATEGORY_DISPLAY.get(cat, cat.replace("_", " ").title()),
        source=raw.get("source", "receipt"),
        notes=raw.get("notes", ""),
        saved_at=raw.get("saved_at", ""),
    )


def get_all_transactions() -> list[TransactionOut]:
    raws = _load_raw()
    return [_to_out(r) for r in raws]


def save_transaction(payload: TransactionCreate) -> TransactionOut:
    transactions = _load_raw()
    next_id = max((t.get("id", 0) for t in transactions), default=0) + 1
    cat = payload.category or "lainnya"
    merchant = payload.merchant.strip()
    try:
        from .extraction_service import clean_merchant_candidate
        merchant = clean_merchant_candidate(merchant) or merchant
    except Exception:
        pass
    record: dict[str, Any] = {
        "id": next_id,
        "merchant": merchant,
        "amount": float(payload.amount),
        "date": payload.date or "",
        "category": cat,
        "source": payload.source or "receipt",
        "notes": payload.notes or "",
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    transactions.append(record)
    _save_raw(transactions)
    return _to_out(record)


def delete_all_transactions() -> dict[str, str]:
    _save_raw([])
    return {"message": "Semua transaksi berhasil dihapus."}
