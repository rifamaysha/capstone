"""Skema unified untuk record transaksi dari 3 sumber data."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class TransactionItem:
    """Satu baris item dalam struk (nama, qty, harga)."""
    name: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total_price: Optional[float] = None


@dataclass
class UnifiedRecord:
    """Record transaksi yang seragam dari sumber manapun.

    Attributes:
        image_path: Path ke gambar PROCESSED (.png hasil preprocessing).
        source: 'kaggle' | 'cord' | 'mbanking'.
        language: 'en' | 'id'.
        merchant: Nama toko/penerima (bisa None untuk Kaggle/CORD).
        category: Label kategori asli dari source (raw, akan di-mapping).
        transaction_date: Tanggal raw string (format bervariasi per source).
        total_amount: Total transaksi dalam currency-nya.
        currency: 'USD' | 'CAD' | 'IDR'.
        items: List item belanja (CORD ada, Kaggle/M-Banking biasanya kosong).
        text_for_classification: Teks gabungan untuk training IndoBERT.
        raw_metadata: Field tambahan dari ground_truth asli.
    """
    image_path: Path
    source: str
    language: str

    merchant: Optional[str] = None
    category: Optional[str] = None
    transaction_date: Optional[str] = None
    total_amount: Optional[float] = None
    currency: Optional[str] = None

    items: list[TransactionItem] = field(default_factory=list)
    text_for_classification: Optional[str] = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize ke dict (Path → str) untuk dump JSON."""
        d = asdict(self)
        d["image_path"] = str(self.image_path)
        return d
