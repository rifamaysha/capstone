"""Parser screenshot M-Banking via OCR + regex (improved heuristics)."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import easyocr

logger = logging.getLogger(__name__)


# Bulan ID + EN + singkatan
_BULAN_PATTERN = (
    r"(Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|"
    r"September|Oktober|November|Desember|"
    r"January|February|March|April|May|June|July|August|"
    r"September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Mei|Jun|Jul|Ags|Agu|Sep|Okt|Nov|Des|May|Aug|Oct|Dec)"
)

_DATE_PATTERNS = [
    # "09 Apr 2026", "24 Februari 2026" — prioritas tinggi (nama bulan)
    re.compile(rf"(\d{{1,2}})\s+{_BULAN_PATTERN}\.?\s+(\d{{4}})", re.IGNORECASE),
    # "April 9, 2026" / "Apr 9 2026"
    re.compile(rf"{_BULAN_PATTERN}\.?\s+(\d{{1,2}}),?\s+(\d{{4}})", re.IGNORECASE),
    # "12/04/2026" "12-04-2026" — TIDAK menerima "." (sering jadi time separator)
    # Year wajib 4 digit untuk hindari "12.14.23" (jam)
    re.compile(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})"),
    # ISO "2026-04-12"
    re.compile(r"(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})"),
]

_AMOUNT_PATTERN = re.compile(
    r"[\-]?\s*(?:Rp\.?|IDR)\s*([\d.,]+)",
    re.IGNORECASE,
)

# Konteks keyword untuk amount — bantu pilih amount yg "benar" di antara banyak
_AMOUNT_CONTEXT_KEYWORDS = [
    "total", "nominal", "jumlah", "amount", "bayar",
    "pembayaran", "transfer", "dibayar", "sukses",
]

_RECIPIENT_KEYWORDS = [
    "penerima", "kepada", "tujuan", "merchant", "nama penerima",
    "recipient", "to:", "ke:", "nama:", "kepada:",
]

# Range plausible: <1000 IDR transfer hampir mustahil; >1M IDR juga jarang utk capstone scope
_MIN_AMOUNT = 1_000
_MAX_AMOUNT = 1_000_000_000


def _is_pan_or_account(text: str) -> bool:
    """True kalau text didominasi digit panjang (10+ digits) — PAN/rekening."""
    digits = re.sub(r"\D", "", text)
    return len(digits) >= 10 and len(digits) / max(len(text), 1) > 0.7


def _has_letters(text: str, min_letters: int = 3) -> bool:
    """True kalau text mengandung cukup huruf (kandidat nama merchant)."""
    letters = sum(1 for c in text if c.isalpha())
    return letters >= min_letters and letters / max(len(text), 1) > 0.3


def _extract_amounts(text: str) -> list[int]:
    """Cari pola 'Rp NNN' / 'IDR NNN' → list integer plausible."""
    text = _normalize_ocr_digits(text)   # NEW: handle OCR O/o → 0
    out: list[int] = []
    for raw in _AMOUNT_PATTERN.findall(text):
        cleaned = _strip_idr_decimal(raw)             # NEW: hapus ',00' decimal
        cleaned = re.sub(r"[.,]", "", cleaned)
        if cleaned.isdigit():
            n = int(cleaned)
            if _MIN_AMOUNT <= n <= _MAX_AMOUNT:
                out.append(n)
    return out


def _extract_amounts_loose(text: str) -> list[int]:
    """Cari angka standalone (tanpa Rp/IDR prefix). Pakai HANYA dengan
    konteks keyword untuk hindari false positive (tanggal, no telp, dll.)."""
    text = _normalize_ocr_digits(text)
    out: list[int] = []
    # Pattern: angka dengan optional separator ribuan + optional decimal
    for m in re.finditer(r"\b(\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?)\b", text):
        cleaned = _strip_idr_decimal(m.group(1))
        cleaned = re.sub(r"[.,]", "", cleaned)
        if cleaned.isdigit():
            n = int(cleaned)
            if _MIN_AMOUNT <= n <= _MAX_AMOUNT:
                out.append(n)
    return out

# Label umum di UI bank — jangan dipakai sebagai recipient
_LABEL_BLACKLIST = {
    "customer pan", "merchant pan", "reference number", "terminal id",
    "acquirer name", "source of fund", "main pocket", "transaction id",
    "transaction details", "date and time", "fee", "category", "share",
    "rincian transaksi", "metode pembayaran", "id transaksi", "id transaksi gojek",
    "bagikan resi", "lihat resi", "kategori", "no. ref", "tipe", "status",
    "selesai", "success", "free", "split", "tutup", "no. ref blu",
    "transaksi", "qris", "atur jumlah", "bagi bukti bayar",
}


def _normalize_ocr_digits(text: str) -> str:
    """Substitusi 'O'/'o' → '0' dalam konteks angka.

    OCR di font aplikasi banking sering salah-baca '0' sebagai 'O'/'o'.
    Kita normalisasi:
    - 'O'/'o' yang dikelilingi/didahului/diikuti digit
    - Runtutan 'O'/'o' setelah pemisah ribuan (contoh: '13.OOO' → '13.000')
    """
    text = re.sub(r"(?<=\d)[Oo]", "0", text)
    text = re.sub(r"[Oo](?=\d)", "0", text)
    text = re.sub(
        r"([.,])([Oo]+)",
        lambda m: m.group(1) + "0" * len(m.group(2)),
        text,
    )
    return text


def _strip_idr_decimal(amount_str: str) -> str:
    """Hapus trailing ',XX' (decimal cents IDR).

    'Rp 69.000,00' → '69.000' → diolah jadi 69000.
    Pemisah ribuan IDR adalah titik; koma dengan 1-2 digit di akhir
    adalah sen yang dapat diabaikan.
    """
    return re.sub(r",\d{1,2}\s*$", "", amount_str.strip())


def _is_label_line(text: str) -> bool:
    """Cek apakah line adalah label UI, bukan value."""
    lower = text.strip().lower()
    if lower in _LABEL_BLACKLIST:
        return True
    return any(lab == lower or lab in lower.split() for lab in _LABEL_BLACKLIST)
    
class MBankingParser:
    """OCR + regex extractor untuk screenshot M-Banking."""

    def __init__(self, languages: tuple[str, ...] = ("en", "id")) -> None:
        logger.info("Initializing EasyOCR (lang=%s, gpu=False)…", languages)
        self.reader = easyocr.Reader(list(languages), gpu=False, verbose=False)

    def extract_text_lines(self, image_path: str | Path) -> list[str]:
        """OCR satu gambar → list of text lines."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        results = self.reader.readtext(str(image_path), detail=0, paragraph=False)
        return [r.strip() for r in results if r.strip()]

    def _parse_amount(self, lines: list[str]) -> float | None:
        """Cari nominal — prioritas dekat keyword, fallback global max."""
        for i, line in enumerate(lines):
            if any(kw in line.lower() for kw in _AMOUNT_CONTEXT_KEYWORDS):
                # Same line: pakai pattern strict (dengan Rp/IDR)
                same = _extract_amounts(line)
                if same:
                    return float(max(same))
                # Look ahead 2 lines: coba strict dulu, lalu loose
                for j in range(i + 1, min(i + 3, len(lines))):
                    strict = _extract_amounts(lines[j])
                    if strict:
                        return float(max(strict))
                    loose = _extract_amounts_loose(lines[j])
                    if loose:
                        return float(max(loose))

        # Fallback global — cuma yang strict
        all_amounts = _extract_amounts("\n".join(lines))
        return float(max(all_amounts)) if all_amounts else None

    def _parse_date(self, text: str) -> str | None:
        """Cari date pertama yang cocok pola — pattern dengan nama bulan diutamakan."""
        text = _normalize_ocr_digits(text)   # NEW: handle OCR 'O' → '0' di tahun
        for pattern in _DATE_PATTERNS:
            m = pattern.search(text)
            if m:
                return m.group(0)
        return None

    def _parse_recipient(self, lines: list[str]) -> str | None:
        """Recipient — keyword first, lalu fallback ke first prominent text.

        Strategy 1: cari keyword 'Penerima'/'Merchant'/dll → ambil baris setelahnya.
        Strategy 2 (NEW): kalau tidak ada keyword (mis. GoPay UI), cari baris
        pertama yang text-rich (bukan status bar, bukan amount, bukan label).
        """
        # Strategy 1: keyword-anchored (logika lama)
        for i, line in enumerate(lines):
            if any(kw in line.lower() for kw in _RECIPIENT_KEYWORDS):
                for j in range(i + 1, min(i + 4, len(lines))):
                    cand = lines[j].strip()
                    if not cand:
                        continue
                    if _is_pan_or_account(cand):
                        continue
                    if _is_label_line(cand):
                        continue
                    if any(kw in cand.lower() for kw in _RECIPIENT_KEYWORDS):
                        continue
                    if not _has_letters(cand):
                        continue
                    return cand

        # Strategy 2: fallback — first prominent text-rich line
        # Skip baris yang adalah: jam, status bar, amount, label, junk
        skip_re = re.compile(
            r"^\s*("
            r"[\d:.,\s]+(wib|wita|wit|am|pm|kb/s|mb/s)?|"
            r"[\-]?\s*rp\s*[\d.,]+|"
            r"qris|qr\s*bayar|"
            r"pembayaran(\s+berhasil)?[!.]?|transaksi(\s+berhasil)?|"
            r"detail\s+transaksi|x|0|@"
            r")\s*$",
            re.IGNORECASE,
        )
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if skip_re.match(stripped):
                continue
            if _is_pan_or_account(stripped):
                continue
            if _is_label_line(stripped):
                continue
            if not _has_letters(stripped, min_letters=4):
                continue
            # Baris pertama yang lolos semua filter = kandidat merchant
            return stripped

        return None

    def parse(
        self,
        image_path: str | Path,
        return_raw: bool = False,
    ) -> dict[str, Any]:
        """Parse screenshot M-Banking → dict {amount, date, recipient, ...}.

        Args:
            image_path: Path ke gambar.
            return_raw: Jika True, sertakan field 'raw_text' untuk debugging.
        """
        lines = self.extract_text_lines(image_path)
        full_text = "\n".join(lines)

        result: dict[str, Any] = {
            "amount":    self._parse_amount(lines),
            "date":      self._parse_date(full_text),
            "recipient": self._parse_recipient(lines),
            "n_lines":   len(lines),
        }
        if return_raw:
            result["raw_text"] = full_text
        return result