"""Parser screenshot M-Banking via OCR + regex (improved heuristics)."""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - optional OCR enhancement
    cv2 = None
    np = None

try:
    import easyocr
except ImportError:
    easyocr = None

logger = logging.getLogger(__name__)

DEMO_FAST_OCR = True
OCR_MAX_DIMENSION = 1000 if DEMO_FAST_OCR else 1200
OCR_MAX_PASSES = 1 if DEMO_FAST_OCR else 2
OCR_SOFT_TIMEOUT_SECONDS = 28.0


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
    re.compile(rf"(\d{{1,2}})\s+{_BULAN_PATTERN}\.?\s+(\d{{2,4}})", re.IGNORECASE),
    # "April 9, 2026" / "Apr 9 2026"
    re.compile(rf"{_BULAN_PATTERN}\.?\s+(\d{{1,2}}),?\s+(\d{{2,4}})", re.IGNORECASE),
    # "12/04/2026" "12-04-2026" — TIDAK menerima "." (sering jadi time separator)
    # Year wajib 4 digit untuk hindari "12.14.23" (jam)
    re.compile(r"\b(\d{4})[/\.\-_|](\d{1,2})[/\.\-_|](\d{1,2})\b"),
    re.compile(r"\b(\d{1,2})[/\.\-_|](\d{1,2})[/\.\-_|](\d{2,4})\b"),
]

_MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "mei": "05", "jun": "06", "jul": "07",
    "aug": "08", "agu": "08", "ags": "08", "sep": "09",
    "okt": "10", "oct": "10", "nov": "11", "des": "12", "dec": "12",
}

_DATE_CONTEXT_KEYWORDS = (
    "tanggal", "date", "waktu", "time", "jam", "transaksi", "transaction",
    "payment", "pembayaran", "selesai", "berhasil", "detail", "rincian",
)

_AMOUNT_PATTERN = re.compile(
    r"[\-\u2013\u2014]?\s*(?:Rp\.?|IDR)\s*([0-9OoIlSBb.,\s]+)",
    re.IGNORECASE,
)

# Konteks keyword untuk amount — bantu pilih amount yg "benar" di antara banyak
# High-priority context keywords — amounts near these are the main transaction amount
_AMOUNT_CONTEXT_KEYWORDS = [
    "total", "nominal", "jumlah", "amount", "bayar",
    "pembayaran", "transfer", "dibayar", "sukses", "berhasil",
    "qris",          # QRIS payment screenshots
    "debit",         # debit transaction line
    "kredit",        # credit transaction line
    "transaksi",     # general transaction label
    "tagihan",       # bill label
    "harga",         # price label
    "charge",        # charge label
    "dana",          # GoPay/Dana app amount context
]
# Low-priority context keywords — amounts near these may be fees or balance.
# Used ONLY as last resort before global max fallback to prevent admin fee
# (Rp3.000 near "biaya admin") from overriding main amount (Rp103.000).
_AMOUNT_CONTEXT_KEYWORDS_LOW = [
    "biaya",         # fee/cost — could be admin fee, not main amount
    "saldo",         # balance — shows remaining balance after transaction
]

_RECIPIENT_KEYWORDS = [
    "penerima", "kepada", "tujuan", "merchant", "nama penerima",
    "recipient", "to:", "ke:", "nama:", "kepada:",
    # QRIS / payment app patterns
    "bayar ke", "bayar kepada", "nama toko", "nama merchant",
    "merchant name", "tujuan transfer", "beneficiary",
    "nama usaha", "diterima oleh", "pembayaran ke", "payment to",
]

_MERCHANT_CONTEXT_KEYWORDS = [
    "bayar ke", "bayar kepada", "payment to", "pembayaran ke",
    "merchant", "nama merchant", "nama toko", "nama usaha",
    "penerima", "kepada", "tujuan", "recipient",
]

# Range plausible: <1000 IDR transfer hampir mustahil; >1M IDR juga jarang utk capstone scope
_MIN_AMOUNT = 1_000
_MAX_AMOUNT = 1_000_000_000


# Regex for alphanumeric transaction / reference IDs (e.g. "260104-RWBR-DBAXWB")
_TRANSACTION_ID_RE = re.compile(
    r"^[A-Z0-9]{3,12}-[A-Z0-9]{3,8}-[A-Z0-9]{3,12}$"   # 3-segment hyphenated
    r"|^[A-Z0-9]{3,10}-[A-Z0-9]{3,12}$"                  # 2-segment hyphenated
    r"|^[A-Z]{2,5}\d{8,18}$",                             # prefix + long digit (TXN20240101...)
    re.IGNORECASE,
)


def _is_transaction_id(text: str) -> bool:
    """True kalau text tampak seperti transaction/reference ID alphanumeric.

    Menangkap pola seperti:
    - '260104-RWBR-DBAXWB'  (Bank Jago 3-segment)
    - 'REF-20240115-XYZ'    (generic 3-segment)
    - 'TXN202401011234'     (prefix + timestamp)
    """
    stripped = text.strip()
    # Must not contain spaces (IDs are compact)
    if " " in stripped:
        return False
    # Must contain at least one letter (not just a phone/account number)
    if not any(c.isalpha() for c in stripped):
        return False
    return bool(_TRANSACTION_ID_RE.match(stripped))


def _is_pan_or_account(text: str) -> bool:
    """True kalau text adalah PAN/rekening (10+ digit) ATAU transaction ID alphanumeric."""
    if _is_transaction_id(text):
        return True
    digits = re.sub(r"\D", "", text)
    return len(digits) >= 10 and len(digits) / max(len(text), 1) > 0.7


def _has_letters(text: str, min_letters: int = 3) -> bool:
    """True kalau text mengandung cukup huruf (kandidat nama merchant)."""
    letters = sum(1 for c in text if c.isalpha())
    return letters >= min_letters and letters / max(len(text), 1) > 0.3


def _amount_doc_context(context: str = "", doc_type: str = "") -> bool:
    ctx = f"{context or ''}\n{doc_type or ''}".lower()
    return any(k in ctx for k in ("compact_qr_card", "qr bayar", "pembayaran qr", "pembayaran qris"))


def _normalize_amount_token(
    amount_str: str,
    *,
    currency_context: bool,
    context: str = "",
    doc_type: str = "",
) -> str:
    """Normalize OCR amount text without losing Indonesian thousand groups.

    Rp/IDR amounts use dots as thousands separators in this dataset, so
    ``Rp3.000`` must become ``3000`` rather than ``3``. Decimal cents are
    stripped only when the separator pattern clearly represents cents.
    """
    s = _normalize_ocr_digits(str(amount_str))
    s = re.sub(r"(?i)\b(rp\.?|idr)\b", "", s)
    s = re.sub(r"[\-\u2013\u2014+]", "", s)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"^[^\d]+|[^\d]+$", "", s)
    if not s:
        return ""
    compact_qr_context = _amount_doc_context(context, doc_type)

    if "." in s and "," in s:
        last_dot = s.rfind(".")
        last_comma = s.rfind(",")
        decimal_sep = "." if last_dot > last_comma else ","
        frac = s.split(decimal_sep)[-1]
        if len(frac) in {1, 2}:
            s = s[: max(last_dot, last_comma)]
        return re.sub(r"\D", "", s)

    if "." in s:
        parts = s.split(".")
        if (
            compact_qr_context
            and currency_context
            and len(parts) == 2
            and len(parts[0]) == 3
            and len(parts[1]) == 3
            and parts[1].startswith("00")
            and parts[1] != "000"
        ):
            return re.sub(r"\D", "", parts[0] + parts[1][:2])
        # All post-dot groups are exactly 3 digits → standard IDR thousands
        if len(parts) > 1 and all(len(part) == 3 for part in parts[1:] if part):
            return re.sub(r"\D", "", s)
        # Two-part: left.right
        if len(parts) == 2:
            left, right = parts[0], parts[1]
            if (
                compact_qr_context
                and currency_context
                and len(left) == 3
                and len(right) == 3
                and right.startswith("00")
                and right != "000"
            ):
                return re.sub(r"\D", "", left + right[:2])
            # Standard 3-digit thousands group → "38.000" → 38000
            if len(right) == 3:
                return re.sub(r"\D", "", s)
            # OCR artifact: >3 digits after dot (e.g. "10.00000", "94.80009").
            # In IDR context the first 3 digits form the thousands group;
            # the excess digits are OCR noise — trim them.
            if len(right) > 3 and currency_context:
                return re.sub(r"\D", "", left + right[:3])
            # 1-2 digit right → decimal cents, strip in currency context
            if len(right) in {1, 2}:
                return re.sub(r"\D", "", left)
        if currency_context:
            return re.sub(r"\D", "", s)
        return re.sub(r"\D", "", s)

    if "," in s:
        parts = s.split(",")
        if (
            compact_qr_context
            and currency_context
            and len(parts) == 2
            and len(parts[0]) == 3
            and len(parts[1]) == 3
            and parts[1].startswith("00")
            and parts[1] != "000"
        ):
            return re.sub(r"\D", "", parts[0] + parts[1][:2])
        if len(parts) > 1 and all(len(part) == 3 for part in parts[1:] if part):
            return re.sub(r"\D", "", s)
        if len(parts) == 2 and len(parts[-1]) in {1, 2}:
            return re.sub(r"\D", "", parts[0])
        return re.sub(r"\D", "", s)

    return re.sub(r"\D", "", s)


def _sanitize_ocr_amount(text: str) -> str:
    """Trim OCR-artifact trailing digits after an IDR thousands group.

    EasyOCR sometimes appends repeated zeros or noise digits after a standard
    3-digit thousands group on compact QR cards and bold fonts:
      "10.00000" → "10.000"   (repeated-zero artifact)
      "26.00009" → "26.000"   (noise digit after zeros)
      "94.80009" → "94.800"   (noise digit after valid group)

    Only the LAST dot group is trimmed when it has more than 3 digits.
    Multi-dot chains (1.000.000) are unaffected because each group has ≤ 3 digits.
    """
    def _trim(m: re.Match) -> str:
        return f"{m.group(1)}.{m.group(2)[:3]}"
    return re.sub(r"(\d+)\.(\d{4,})(?!\d*\.)", _trim, text)


def _extract_amounts(text: str, *, context: str = "", doc_type: str = "") -> list[int]:
    """Cari pola 'Rp NNN' / 'IDR NNN' -> list integer plausible."""
    text = _normalize_ocr_digits(_sanitize_ocr_amount(text))
    out: list[int] = []
    for raw in _AMOUNT_PATTERN.findall(text):
        cleaned = _normalize_amount_token(raw, currency_context=True, context=context, doc_type=doc_type)
        if cleaned.isdigit():
            n = int(cleaned)
            if _MIN_AMOUNT <= n <= _MAX_AMOUNT:
                out.append(n)
    return out


def _extract_amounts_loose(text: str, *, context: str = "", doc_type: str = "") -> list[int]:
    """Cari angka standalone (tanpa Rp/IDR prefix). Pakai HANYA dengan
    konteks keyword untuk hindari false positive (tanggal, no telp, dll.)."""
    text = _normalize_ocr_digits(_sanitize_ocr_amount(text))
    out: list[int] = []
    # Pattern: angka dengan optional separator ribuan + optional decimal
    for m in re.finditer(r"\b(\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?)\b", text):
        cleaned = _normalize_amount_token(
            m.group(1),
            currency_context=False,
            context=context,
            doc_type=doc_type,
        )
        if cleaned.isdigit():
            n = int(cleaned)
            if _MIN_AMOUNT <= n <= _MAX_AMOUNT:
                out.append(n)
    return out

# Label umum di UI bank — jangan dipakai sebagai recipient
_LABEL_BLACKLIST = {
    # Field labels (English)
    "customer pan", "merchant pan", "reference number", "terminal id",
    "acquirer name", "source of fund", "main pocket", "transaction id",
    "transaction details", "transaction detail", "date and time", "fee",
    "category", "share", "from", "to", "details", "status",
    "payment details", "payment method", "transfer details",
    "nama pengakuisisi", "acquirer",
    # Category labels that must never become merchant
    "shopping", "groceries", "grocery", "food", "beverage",
    "food & drink", "food and drink", "food & beverage",
    "transport", "transportation", "travel",
    "makanan & minuman", "makanan dan minuman", "makanan", "minuman",
    "belanja", "belanja & retail", "retail", "lainnya", "other",
    "pendidikan", "education", "kesehatan", "health", "healthcare",
    "hiburan", "entertainment", "tagihan", "bills", "utilities",
    # Field labels (Indonesian)
    "rincian transaksi", "metode pembayaran", "id transaksi", "id transaksi gojek",
    "bagikan resi", "lihat resi", "kategori", "no. ref", "tipe",
    "selesai", "success", "free", "split", "tutup", "no. ref blu",
    "transaksi", "qris", "atur jumlah", "bagi bukti bayar",
    "detail transaksi", "rincian pembayaran", "informasi transaksi",
    "sumber dana", "dari rekening", "ke rekening",
    "total", "subtotal", "grand total", "bayar", "nominal",
    "berhasil dikirim", "transfer berhasil", "pembayaran berhasil",
    "kode qr", "qr bayar", "qr code", "scan qr",
    "nama bank", "nama rekening", "no rekening", "nomor rekening",
    "nama merchant",
    "saldo akhir", "biaya admin", "biaya transfer",
    "rincian pesanan", "transaction sn", "order sn", "merchant ref id",
    "merchant location", "total payment", "date", "tanggal", "amount",
    "rincian pembayaran", "pembayaran qr", "pembayaran qris berhasil",
    "qris payment successful", "transaction details",
    # Single tokens that are always UI labels
    "transfer", "debit", "kredit",
}

_BANK_STATUS_NAMES = {
    "bca", "bank bca", "bank jago", "seabank", "bank mandiri", "bni",
    "bri", "btn", "cimb", "permata", "maybank", "ocbc", "danamon",
    "shopeepay", "gopay", "dana", "ovo", "linkaja", "blu",
    "qris payment successful", "pembayaran berhasil", "transaction details",
    "rincian pembayaran", "rincian transaksi",
}

_CITY_ONLY_SET: frozenset[str] = frozenset({
    "bandung", "jakarta", "surabaya", "medan", "semarang", "makassar",
    "denpasar", "malang", "bogor", "depok", "bekasi", "tangerang",
    "bali", "jogja", "yogyakarta", "solo", "palembang", "balikpapan",
    "banjarmasin", "pontianak", "manado", "pekanbaru", "padang",
    "id", "idn", "indonesia",
})


def _normalize_ocr_digits(text: str) -> str:
    """Substitusi karakter OCR yang sering salah-baca dalam konteks angka.

    - 'O'/'o' → '0'  (font banking bulat mirip nol)
    - 'I'/'l' → '1'  (hanya di antara digit, konservatif)
    - 'S'     → '5'  (hanya di antara digit, konservatif)
    """
    text = re.sub(r"(?<=\d)[Oo]", "0", text)
    text = re.sub(r"[Oo](?=\d)", "0", text)
    text = re.sub(
        r"([.,])([Oo]+)",
        lambda m: m.group(1) + "0" * len(m.group(2)),
        text,
    )
    # I/l → 1 hanya kalau diapit digit
    text = re.sub(r"(?<=\d)[Il](?=\d)", "1", text)
    # l/I → 1 kalau didahului huruf dan diikuti digit (mis. 'Rpl05' → 'Rp105')
    text = re.sub(r"(?<=[a-zA-Z])([Il])(?=\d)", "1", text)
    # S → 5 dalam konteks numerik (mis. '10S000' → '105000', '10S.000' → '105.000')
    text = re.sub(r"(?<=\d)S(?=\d)", "5", text)        # antara dua digit
    text = re.sub(r"(?<=\d)S(?=[.,]\d)", "5", text)    # sebelum separator ribuan
    # B → 8 dalam konteks numerik:
    #   'Rp3B.000' → 'Rp38.000'  (B before dot/comma)
    #   '3B000'    → '38000'     (B before digit)
    #   '3B 000'   → '38 000'    (B before space+digit — OCR artifact on large fonts)
    text = re.sub(r"(?<=\d)B(?=[\d.,])", "8", text)
    text = re.sub(r"(?<=\d)B(?=\s\d)", "8", text)
    text = re.sub(r"(?<=\d)b(?=[\d.,])", "8", text)
    text = re.sub(r"(?<=\d)b(?=\s\d)", "8", text)
    return text


def _normalize_date_ocr_text(text: str) -> str:
    text = _normalize_ocr_digits(text or "")
    text = re.sub(r"(\d{1,2}[/.\-_]\d{1,2}[/.\-_]\d{2})\|(\d)", r"\g<1>1\2", text)
    text = re.sub(r"(\d{4}[/.\-_]\d{1,2}[/.\-_]\d)\|(\d)", r"\1\2", text)
    text = re.sub(r"(?<=\d)[Oo](?=\d|[/.\-_|])", "0", text)
    text = re.sub(r"(?<=[/.\-_|])[Oo](?=\d)", "0", text)
    text = re.sub(r"(?<=\d)S(?=\d|[/.\-_|])", "5", text)
    text = re.sub(r"(?<=\d)[Il](?=\d|[/.\-_|])", "1", text)
    text = re.sub(r"(?<=\d)\s*([/.\-_|])\s*(?=\d)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _expand_year(year: str) -> int:
    if len(year) == 2:
        y = int(year)
        return 2000 + y if y <= 35 else 1900 + y
    return int(year)


def _date_value(day: str, month: str, year: str) -> str:
    d, m, y = int(day), int(month), _expand_year(year)
    if m > 12 and 1 <= d <= 12 and m <= 31:
        d, m = m, d
    if not (1 <= d <= 31 and 1 <= m <= 12 and 2010 <= y <= 2035):
        return ""
    try:
        datetime(y, m, d)
    except ValueError:
        return ""
    return f"{d:02d}/{m:02d}/{y:04d}"


def _normalize_date_candidate(raw: str) -> str:
    s = _normalize_date_ocr_text(raw)
    s = re.sub(
        r"(?i)\b(date|tanggal|waktu\s+selesai|date\s+and\s+time|payment\s+date|transaction\s+date|time|jam)\b\s*[:\-]?\s*",
        "",
        s,
    ).strip()
    m = re.match(r"(\d{1,2})\s+([A-Za-z]+)\.?\s+(\d{2,4})", s)
    if m:
        mn = _MONTH_MAP.get(m.group(2)[:3].lower())
        return _date_value(m.group(1), mn, m.group(3)) if mn else ""
    m = re.match(r"([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{2,4})", s)
    if m:
        mn = _MONTH_MAP.get(m.group(1)[:3].lower())
        return _date_value(m.group(2), mn, m.group(3)) if mn else ""
    m = re.match(r"(\d{4})[/.\-_|](\d{1,2})[/.\-_|](\d{1,2})", s)
    if m:
        return _date_value(m.group(3), m.group(2), m.group(1))
    m = re.match(r"(\d{1,2})[/.\-_|](\d{1,2})[/.\-_|](\d{2,4})", s)
    if m:
        return _date_value(m.group(1), m.group(2), m.group(3))
    return ""


def _date_candidates(text: str) -> list[dict[str, Any]]:
    lines = [_normalize_date_ocr_text(line) for line in (text or "").splitlines() if line.strip()]
    if not lines:
        lines = [_normalize_date_ocr_text(text or "")]
    candidates: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        window = " ".join(lines[max(0, i - 1): min(len(lines), i + 2)]).lower()
        for pidx, pattern in enumerate(_DATE_PATTERNS):
            for m in pattern.finditer(line):
                raw = m.group(0)
                value = _normalize_date_candidate(raw)
                if not value:
                    continue
                score = 60
                reason = [f"pattern_{pidx}"]
                if any(k in window for k in _DATE_CONTEXT_KEYWORDS):
                    score += 50
                    reason.append("date_context")
                if i <= max(5, len(lines) // 2):
                    score += 20
                    reason.append("upper_detail")
                if re.search(r"\b\d{1,2}[:.]\d{2}(?::\d{2})?\b", line):
                    score += 8
                    reason.append("near_time")
                if re.search(r"\b(transaction id|reference|no\.?\s*ref|pan|terminal|rrn|stan)\b", window):
                    score -= 70
                    reason.append("id_context_penalty")
                candidates.append({
                    "raw": raw,
                    "normalized": value,
                    "line": i,
                    "score": score,
                    "reason": "|".join(reason),
                })
    dedup: dict[tuple[str, int], dict[str, Any]] = {}
    for cand in candidates:
        key = (cand["normalized"], cand["line"])
        if key not in dedup or cand["score"] > dedup[key]["score"]:
            dedup[key] = cand
    return sorted(dedup.values(), key=lambda c: (c["score"], -c["line"]), reverse=True)


def _strip_idr_decimal(amount_str: str) -> str:
    """Hapus trailing decimal cents dari string amount.

    Handles dua gaya desimal:
    - IDR style  : '69.000,00' → '69.000'  (koma+1-2 digit di akhir)
    - US style   : '105,000.00' → '105,000' (titik+1-2 digit di akhir)

    Hanya strip kalau trailing part tepat 1-2 digit (bukan ribuan '000').
    """
    s = re.sub(r"\s+", "", amount_str.strip())
    s = re.sub(r"\.\d{1,2}\s*$", "", s)   # .00 / .0  (US decimal)
    s = re.sub(r",\d{1,2}\s*$", "", s)    # ,00 / ,0  (IDR decimal)
    return s


def _clean_inline_merchant(raw: str) -> str:
    clean = re.sub(r"\s+", " ", raw).strip(" :;-")
    # Require whitespace after bare "ke"/"to" so "Keikpop" / "Tokyo" are not stripped
    clean = re.sub(r"^(?:ke|to)\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^(?:merchant|nama\s+merchant|bayar\s+ke)\s*[:\-]?\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r",\s*\d{6,}.*$", "", clean)
    clean = re.sub(r"\s+\d{8,}.*$", "", clean)
    clean = re.split(
        r"\s*(?:;|\||/|\s-\s)\s*(?=(?:jl\.?|jalan|ruko|komplek|perumahan|blok|rt\.?|rw\.?|"
        r"kel\.?|kec\.?|kecamatan|kelurahan|kota|kab\.?|kabupaten|kode\s+pos|no\.?|lt\.?|lantai)\b)",
        clean,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    clean = re.sub(
        r",\s*(babakan|bandung|jakarta|surabaya|kota|kab\.?).*$",
        "",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\s*,\s*(?:jl\.?|jalan|ruko|komplek|perumahan|blok|rt\.?|rw\.?|kel\.?|kec\.?|"
        r"kecamatan|kelurahan|kota|kab\.?|kabupaten|kode\s+pos|no\.?\s+\d|lt\.?|lantai)\b.*$",
        "",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\s+\b(?:jl\.?|jalan|rt\.?|rw\.?|kec\.?|kel\.?|kecamatan|kelurahan|kode\s+pos)\s+[A-Za-z0-9].*$",
        "",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\s*,\s*(?:jakarta|bandung|bekasi|tangerang|surabaya|depok|bogor|semarang|"
        r"yogyakarta|jogja|medan|malang|denpasar|bali)(?:\s+(?:barat|timur|utara|selatan|pusat))?\b.*$",
        "",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(r",\s*\d{4,6}(?:\s*,?\s*[A-Z]{2})?\s*$", "", clean)
    clean = clean.strip(" ,.;:-")
    if len(clean) > 55:
        shortened = re.sub(r"\s+\S*$", "", clean[:55]).strip(" ,.;:-")
        if len(shortened) >= 8:
            clean = shortened
    return clean


def _normalize_service_merchant(text: str) -> str:
    lower = text.lower()
    if "gobills" in lower and "pln" in lower:
        return "GoBills PLN Token"
    if "pln token" in lower:
        return "PLN Token"
    return text.strip()


def _looks_like_bad_merchant(text: str) -> bool:
    stripped = text.strip()
    lower = stripped.lower()
    if not stripped:
        return True
    if lower in _BANK_STATUS_NAMES:
        return True
    if any(lower.startswith(prefix) for prefix in ("jl.", "jl ", "jalan", "alamat", "ruko", "komplek")):
        return True
    # Reject category labels
    if lower in _LABEL_BLACKLIST:
        return True
    # Reject city-only strings
    if lower in _CITY_ONLY_SET:
        return True
    if lower in {
        "bca", "bank bca", "bank jago", "seabank", "bank mandiri",
        "shopeepay", "gopay", "dana", "ovo", "success", "berhasil",
        "selesai", "detail", "salin", "share", "free", "qris", "qr",
        "transaction details", "rincian pembayaran", "rincian pesanan",
    }:
        return True
    if _is_pan_or_account(stripped) or _is_transaction_id(stripped):
        return True
    if re.match(r"^[A-Z0-9]{3,12}(?:-[A-Z0-9]{2,12}){1,4}$", stripped, re.IGNORECASE):
        if any(c.isalpha() for c in stripped) and any(c.isdigit() for c in stripped):
            return True
    if _extract_amounts(stripped) or re.match(r"^[\-\u2013\u2014]?\s*(rp|idr)\b", lower):
        return True
    if re.search(r"\b(rp|idr)\s*[\d.,oO]+\b", lower):
        return True
    if re.search(r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}", stripped):
        return True
    if _date_candidates(stripped):
        return True
    if re.fullmatch(
        rf"\d{{1,2}}\s+{_BULAN_PATTERN}\.?\s+\d{{2,4}}",
        stripped,
        re.IGNORECASE,
    ):
        return True
    if re.search(r"\b\d{1,2}[:.]\d{2}(:\d{2})?\b", stripped):
        return True
    if re.match(r"^[A-Z][A-Z\s]+,\s*\d{4,6}\s*,\s*[A-Z]{2}$", stripped, re.IGNORECASE):
        return True
    if len(re.sub(r"\D", "", stripped)) >= 6 and not re.search(r"[A-Za-z]{4,}", stripped):
        return True
    if re.match(r"^[A-Z0-9]{6,}$", stripped, re.I) and any(c.isalpha() for c in stripped) and any(c.isdigit() for c in stripped):
        return True
    if re.match(r"^[A-Z]{1,3}\d{1,4}$", stripped, re.I):
        return True
    # Phone status-bar / network-speed text: "0.2KBIs%", "33.3 KB/s", "89%"
    if re.search(
        r"\b(kb/?s|kbis|mb/?s|mbps)\b"    # network speed units
        r"|\d+\s*%\s*$"                     # percentage at end of string
        r"|^\d[\d.,]+\s*[kK][bB]",         # leading digit + KB prefix
        stripped,
        re.IGNORECASE,
    ):
        return True
    # Numeric prefix + QR/payment label: "89 QR Bayar", "12 Pembayaran"
    if re.match(r"^\d+\s+", stripped) and re.search(
        r"\b(qr\s+bayar|pembayaran|berhasil)\b", lower
    ):
        return True
    letters = [c for c in stripped if c.isalpha()]
    if len(letters) >= 5:
        vowels = sum(c.lower() in "aeiou" for c in letters)
        if vowels <= 1:
            return True
    if len(stripped) > 65 and re.search(
        r"\b(jl\.?|jalan|rt\.?|rw\.?|kec\.?|kel\.?|kota|kab\.?|kode\s+pos|no\.?)\b",
        lower,
    ):
        return True
    return _is_label_line(stripped)


def _is_label_line(text: str) -> bool:
    """True if line is a UI label — not a merchant/recipient value.

    Matching rules:
    - Exact full-line match against blacklist
    - Single-word blacklist entry: word-token match (prevents 'fee' matching 'coffee')
    - Multi-word blacklist entry: substring match in lowercase line
    """
    lower = text.strip().lower()
    if not lower:
        return False
    if lower in _LABEL_BLACKLIST:
        return True
    words = lower.split()
    for lab in _LABEL_BLACKLIST:
        lab_words = lab.split()
        if len(lab_words) == 1:
            # Single-word: whole-word token match ONLY for short lines (<= 2 words).
            # Prevents rejecting valid merchants that BEGIN with a label token,
            # e.g. real merchant names containing "QRIS" should NOT be rejected just because
            # 'qris' appears in the blacklist.
            if lab in words and len(words) <= 2:
                return True
        else:
            # Multi-word: substring match in the line (catches 'qr bayar' in '89 QR Bayar')
            if lab in lower:
                return True
    return False


def _merge_split_amounts(lines: list[str]) -> list[str]:
    """Rejoin amount numbers that EasyOCR splits across adjacent OCR blocks.

    EasyOCR sometimes breaks bold large text like '-Rp105.000' into separate
    blocks: ['-Rp105', '.000'] or ['Rp', '105.000']. Merge these before parsing.
    """
    result: list[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i].strip()
        if i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            # Case A: 'Rp105' + '.000' or ',000' → 'Rp105.000' (split at thousand sep)
            if re.search(r"\d$", cur) and re.match(r"^[.,]\d{2,3}$", nxt):
                logger.debug("merge_split_amounts: %r + %r", cur, nxt)
                result.append(cur + nxt)
                i += 2
                continue
            # Case A2: '-Rp3' + '000' -> '-Rp3.000' when the second OCR
            # block is exactly a thousand group without the separator.
            if re.search(r"(?:Rp|IDR)?\s*\d{1,3}$", cur, re.IGNORECASE) and re.match(r"^\d{3}$", nxt):
                logger.debug("merge_split_amounts: %r + %r", cur, nxt)
                result.append(cur + "." + nxt)
                i += 2
                continue
            # Case B: '-Rp' or 'Rp' alone (possibly with leading dash) + '105.000'
            if re.search(r"(?:Rp|IDR)\s*$", cur, re.IGNORECASE) and re.match(r"^[\d\-–—]", nxt):
                logger.debug("merge_split_amounts: %r + %r", cur, nxt)
                result.append(cur + nxt)
                i += 2
                continue
        result.append(cur)
        i += 1
    return result


# Map visible category labels in banking app UIs → internal category keys
_SCREENSHOT_CATEGORY_MAP: dict[str, str] = {
    # English labels (Bank Jago, SeaBank, etc.)
    "shopping":        "belanja",
    "groceries":       "makanan_minuman",
    "grocery":         "makanan_minuman",
    "food":            "makanan_minuman",
    "food & drink":    "makanan_minuman",
    "food and drink":  "makanan_minuman",
    "food & beverage": "makanan_minuman",
    "restaurant":      "makanan_minuman",
    "transport":       "transportasi",
    "transportation":  "transportasi",
    "travel":          "transportasi",
    "bill":            "tagihan",
    "bills":           "tagihan",
    "utilities":       "tagihan",
    "entertainment":   "hiburan",
    "health":          "kesehatan",
    "healthcare":      "kesehatan",
    "education":       "pendidikan",
    # Indonesian labels
    "belanja":         "belanja",
    "makanan":         "makanan_minuman",
    "makan & minum":   "makanan_minuman",
    "makanan & minuman": "makanan_minuman",
    "transportasi":    "transportasi",
    "tagihan":         "tagihan",
    "hiburan":         "hiburan",
    "kesehatan":       "kesehatan",
    "pendidikan":      "pendidikan",
}


def detect_screenshot_category(lines: list[str]) -> str | None:
    """Return internal category key if a known category label is visible in OCR lines."""
    label_markers = ("category", "kategori", "type", "tipe")
    for i, line in enumerate(lines):
        lower = re.sub(r"\s+", " ", line.strip().lower())
        cat = _SCREENSHOT_CATEGORY_MAP.get(lower)
        if cat:
            logger.debug("detect_screenshot_category: %r → %r", line.strip(), cat)
            return cat
        if any(marker in lower for marker in label_markers):
            inline = re.sub(r"^(category|kategori|type|tipe)\s*[:\-]?\s*", "", lower)
            cat = _SCREENSHOT_CATEGORY_MAP.get(inline)
            if cat:
                return cat
            for j in range(i + 1, min(i + 3, len(lines))):
                next_lower = re.sub(r"\s+", " ", lines[j].strip().lower())
                cat = _SCREENSHOT_CATEGORY_MAP.get(next_lower)
                if cat:
                    return cat
    return None


class MBankingParser:
    """OCR + regex extractor untuk screenshot M-Banking."""

    def __init__(self, languages: tuple[str, ...] = ("en", "id")) -> None:
        logger.info("Initializing EasyOCR (lang=%s, gpu=False)…", languages)
        if easyocr is None:
            raise ImportError("easyocr is required to instantiate MBankingParser")
        self.reader = easyocr.Reader(list(languages), gpu=False, verbose=False)
        self.last_timing: dict[str, float | int] = {}

    def extract_text_lines(self, image_path: str | Path) -> list[str]:
        """OCR satu gambar → list of text lines."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self._readtext_with_preprocessing(image_path)

    def _readtext_with_preprocessing(self, image_path: Path) -> list[str]:
        """Run OCR with bounded preprocessing for demo-safe latency."""
        started_at = time.perf_counter()
        timing: dict[str, float | int] = {
            "load_image": 0.0,
            "preprocess": 0.0,
            "ocr": 0.0,
            "ocr_pass_1": 0.0,
            "ocr_passes": 0,
        }

        def clean_lines(results: list[Any]) -> list[str]:
            return [str(r).strip() for r in results if str(r).strip()]

        def evidence_score(lines: list[str]) -> int:
            text = "\n".join(lines).lower()
            score = len(lines)
            score += 8 * len(re.findall(r"\b(rp|idr)\s*[\d.,oOIlSBb]{4,}", text))
            score += 5 * sum(kw in text for kw in (
                "transaction details", "rincian transaksi", "qr bayar", "pembayaran qr",
                "source of fund", "merchant pan", "customer pan", "terminal id",
                "total", "nominal", "jumlah", "bayar ke", "payment to",
            ))
            if re.search(r"\bke\s+[a-z][a-z0-9 .,&'-]{2,}", text):
                score += 8
            return score

        def trim_blank_margins(img: Any) -> Any:
            """Conservatively trim large blank margins on tall screenshots."""
            if cv2 is None or np is None or img is None:
                return img
            h, w = img.shape[:2]
            if h < 1600 or h / max(w, 1) < 1.8:
                return img
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            non_blank = gray < 248
            coords = cv2.findNonZero(non_blank.astype("uint8"))
            if coords is None:
                return img
            x, y, bw, bh = cv2.boundingRect(coords)
            pad = max(24, int(min(h, w) * 0.04))
            y0 = max(0, y - pad)
            y1 = min(h, y + bh + pad)
            x0 = max(0, x - pad)
            x1 = min(w, x + bw + pad)
            if (y1 - y0) < h * 0.92 and (y1 - y0) > h * 0.35 and (x1 - x0) > w * 0.45:
                return img[y0:y1, x0:x1]
            return img

        variants: list[Any] = [str(image_path)]
        if cv2 is not None and np is not None:
            load_started = time.perf_counter()
            img = cv2.imread(str(image_path))
            timing["load_image"] = time.perf_counter() - load_started
            if img is not None:
                prep_started = time.perf_counter()
                img = trim_blank_margins(img)
                h, w = img.shape[:2]
                longest = max(h, w)
                if longest > OCR_MAX_DIMENSION:
                    scale = OCR_MAX_DIMENSION / longest
                    base = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    variants[0] = base
                else:
                    scale = 1.0 if DEMO_FAST_OCR else (min(2.0, OCR_MAX_DIMENSION / longest) if longest < OCR_MAX_DIMENSION else 1.0)
                    base = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1.01 else img
                gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
                denoised = cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
                blur = cv2.GaussianBlur(clahe, (0, 0), 1.0)
                if not DEMO_FAST_OCR:
                    variants.append(cv2.addWeighted(clahe, 1.45, blur, -0.45, 0))
                timing["preprocess"] = time.perf_counter() - prep_started

        ocr_started = time.perf_counter()
        best_lines = clean_lines(self.reader.readtext(variants[0], detail=0, paragraph=False))
        timing["ocr_pass_1"] = time.perf_counter() - ocr_started
        timing["ocr"] = timing["ocr_pass_1"]
        timing["ocr_passes"] = int(timing["ocr_passes"]) + 1
        best_score = evidence_score(best_lines)
        if best_score >= 28:
            timing["total"] = time.perf_counter() - started_at
            timing["best_score"] = best_score
            self.last_timing = timing
            logger.info("OCR timing %s", {k: round(v, 3) if isinstance(v, float) else v for k, v in timing.items()})
            return best_lines

        if DEMO_FAST_OCR:
            timing["total"] = time.perf_counter() - started_at
            timing["best_score"] = best_score
            self.last_timing = timing
            logger.info("OCR timing %s", {k: round(v, 3) if isinstance(v, float) else v for k, v in timing.items()})
            return best_lines

        for variant in variants[1:OCR_MAX_PASSES]:
            if time.perf_counter() - started_at >= OCR_SOFT_TIMEOUT_SECONDS:
                logger.info("OCR soft guard skipped extra preprocessing variant after %.2fs", time.perf_counter() - started_at)
                break
            try:
                ocr_started = time.perf_counter()
                lines = clean_lines(self.reader.readtext(variant, detail=0, paragraph=False))
                pass_elapsed = time.perf_counter() - ocr_started
                timing[f"ocr_pass_{int(timing['ocr_passes']) + 1}"] = pass_elapsed
                timing["ocr"] = float(timing["ocr"]) + pass_elapsed
                timing["ocr_passes"] = int(timing["ocr_passes"]) + 1
            except Exception as exc:  # noqa: BLE001
                logger.debug("OCR preprocessing variant failed: %s", exc)
                continue
            score = evidence_score(lines)
            if score > best_score:
                best_lines, best_score = lines, score
            if score >= 35:
                break
        timing["total"] = time.perf_counter() - started_at
        timing["best_score"] = best_score
        self.last_timing = timing
        logger.info("OCR timing %s", {k: round(v, 3) if isinstance(v, float) else v for k, v in timing.items()})
        return best_lines

    def _parse_amount_with_trace(self, lines: list[str]) -> tuple[float | None, list[dict[str, Any]]]:
        """Score screenshot amount candidates by context and prominence."""
        candidates: list[dict[str, Any]] = []
        n_lines = max(len(lines), 1)
        context_text = "\n".join(lines).lower()
        doc_type = "compact_qr_card" if _amount_doc_context(context_text, "") and len(lines) <= 20 else ""
        has_pln_service = "pln" in context_text or "gobills" in context_text
        has_success_context = any(
            kw in context_text for kw in (
                "berhasil", "sukses", "payment successful", "pembayaran qris",
                "qr bayar", "rincian pembayaran", "transaction details",
            )
        )

        def add(line_idx: int, value: int, score: int, reason: str, line_text: str) -> None:
            if not (_MIN_AMOUNT <= value <= _MAX_AMOUNT):
                return
            lower = line_text.lower()
            norm_lower = _normalize_ocr_digits(lower)
            if any(k in lower for k in ("biaya admin", "fee", "saldo")):
                score -= 55
                reason += "|fee_or_balance_penalty"
            if re.search(
                r"\b(id transaksi|transaction id|reference|referensi|no\.?\s*ref|"
                r"merchant pan|customer pan|terminal id|token|source of fund|"
                r"sumber dana|rekening|account|phone|telepon|tanggal|date|time|"
                r"rrn|stan|trace|auth|approval|invoice|order\s*(?:sn|no|id)|"
                r"transaction\s*sn|merchant\s*ref)\b",
                lower,
            ):
                score -= 90
                reason += "|id_token_penalty"
            if "token" in lower and not re.search(r"\b(rp|idr)\b", norm_lower):
                score -= 90
                reason += "|token_number_penalty"
            if re.search(r"^[\s\-–—]*(?:rp|idr)\b", norm_lower):
                score += 45
                reason += "|currency_line"
            if re.search(r"[\-–—]\s*(?:rp|idr)\b", norm_lower):
                score += 30
                reason += "|outgoing_negative"
            if line_idx <= max(3, n_lines // 3):
                score += 35
                reason += "|top_prominent"
            if line_idx <= max(5, n_lines // 2) and has_success_context:
                score += 18
                reason += "|upper_success_area"
            if any(k in lower for k in ("total", "nominal", "jumlah", "amount", "tagihan")):
                score += 55
                reason += "|main_amount_keyword"
            if has_pln_service and value >= 50_000:
                score += 55
                reason += "|pln_main_amount"
            if any(k in lower for k in ("berhasil", "sukses", "payment successful", "pembayaran qris", "qr bayar", "rincian pembayaran")):
                score += 35
                reason += "|success_context"
            candidates.append({
                "line": line_idx,
                "text": line_text,
                "value": value,
                "score": score,
                "reason": reason,
            })

        for i, line in enumerate(lines):
            line_norm = _normalize_ocr_digits(line)
            for value in _extract_amounts(line_norm, context=context_text, doc_type=doc_type):
                add(i, value, 90, "currency_same_line", line)

            lower = line_norm.lower()
            if any(kw in lower for kw in _AMOUNT_CONTEXT_KEYWORDS):
                for j in range(i + 1, min(i + 4, len(lines))):
                    next_line = _normalize_ocr_digits(lines[j])
                    values = (
                        _extract_amounts(next_line, context=context_text, doc_type=doc_type)
                        or _extract_amounts_loose(next_line, context=context_text, doc_type=doc_type)
                    )
                    for value in values:
                        add(j, value, 70, f"near_keyword_line[{i}]", lines[j])
                    if values:
                        break

        if not candidates:
            full_text = "\n".join(lines)
            for value in _extract_amounts(full_text, context=context_text, doc_type=doc_type):
                add(0, value, 45, "global_currency_fallback", full_text[:120])
            if not candidates:
                for value in _extract_amounts_loose(full_text, context=context_text, doc_type=doc_type):
                    add(0, value, 25, "global_loose_fallback", full_text[:120])

        if not candidates:
            return None, []

        # If a screen has a clear currency amount in the prominent upper area,
        # do not let smaller admin fees or IDs win on a tie.
        prominent_currency = [
            c for c in candidates
            if c["line"] <= max(6, n_lines // 2)
            and re.search(r"\b(rp|idr)\b", _normalize_ocr_digits(c["text"]), re.I)
            and "fee_or_balance_penalty" not in c["reason"]
            and "id_token_penalty" not in c["reason"]
        ]
        if prominent_currency:
            for c in prominent_currency:
                c["score"] += 20
                c["reason"] += "|prominent_currency_bonus"

        best = max(candidates, key=lambda c: (c["score"], c["value"]))
        logger.debug("_parse_amount candidates=%s selected=%s", candidates, best)
        if best["score"] < 45:
            return None, candidates
        return float(best["value"]), candidates

    def _parse_amount(self, lines: list[str]) -> float | None:
        """Cari nominal — collect ALL keyword-adjacent amounts, return MAX.

        Two-tier keyword approach: high-priority keywords (total, nominal, bayar, ...)
        are searched first. Low-priority keywords (biaya, saldo) are only consulted
        if no high-priority amount is found. This prevents the GoBills admin fee
        (Rp3.000 near 'biaya admin') from overriding the main amount (Rp103.000
        near 'tagihan'/'nominal').
        """
        context_text = "\n".join(lines).lower()
        doc_type = "compact_qr_card" if _amount_doc_context(context_text, "") and len(lines) <= 20 else ""

        def _collect_near_keywords(kw_list: list[str]) -> list[int]:
            collected: list[int] = []
            for i, line in enumerate(lines):
                if not any(kw in line.lower() for kw in kw_list):
                    continue
                # Same line: strict (Rp/IDR prefix)
                same = _extract_amounts(line, context=context_text, doc_type=doc_type)
                if same:
                    collected.extend(same)
                else:
                    # Look ahead up to 2 lines for the value
                    for j in range(i + 1, min(i + 3, len(lines))):
                        strict = _extract_amounts(lines[j], context=context_text, doc_type=doc_type)
                        if strict:
                            collected.extend(strict)
                            break
                        loose = _extract_amounts_loose(lines[j], context=context_text, doc_type=doc_type)
                        if loose:
                            collected.extend(loose)
                            break
            return collected

        # Tier 1: high-priority keywords (main transaction amount)
        hi = _collect_near_keywords(_AMOUNT_CONTEXT_KEYWORDS)
        if hi:
            best = float(max(hi))
            logger.debug("_parse_amount tier-1 candidates=%s → %s", hi, best)
            return best

        # Tier 2: low-priority keywords (fees / balance — last resort)
        lo = _collect_near_keywords(_AMOUNT_CONTEXT_KEYWORDS_LOW)
        if lo:
            best = float(max(lo))
            logger.debug("_parse_amount tier-2 (low-pri) candidates=%s → %s", lo, best)
            return best

        # Fallback global — strict (Rp/IDR prefix) first, then loose (thousand-separator)
        full_text = "\n".join(lines)
        all_strict = _extract_amounts(full_text, context=context_text, doc_type=doc_type)
        if all_strict:
            return float(max(all_strict))
        all_loose = _extract_amounts_loose(full_text, context=context_text, doc_type=doc_type)
        return float(max(all_loose)) if all_loose else None

    def _parse_date_scored(self, text: str) -> str | None:
        candidates = _date_candidates(text)
        return candidates[0]["normalized"] if candidates else None

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

        Strategy 0: "Pembayaran QR ke <merchant>" / "QR Bayar ke <merchant>" inline.
        Strategy 1: cari keyword 'Penerima'/'Merchant'/dll → ambil baris setelahnya.
        Strategy 2: fallback — first text-rich line that is not a UI label / amount.
        """
        def good_candidate(raw: str) -> str | None:
            clean = _clean_inline_merchant(raw)
            clean = _normalize_service_merchant(clean)
            if len(clean) < 2 or _looks_like_bad_merchant(clean):
                return None
            return clean

        # ------------------------------------------------------------------ #
        # Strategy 0: inline QR-payment pattern                               #
        # "Pembayaran QR ke Pasta Nafisa, ..." → "Pasta Nafisa"              #
        # ------------------------------------------------------------------ #
        for line in lines:
            m = re.search(
                r"(?:pembayaran\s+qr|qr\s+bayar)\s+ke\s+(.+)",
                line,
                re.IGNORECASE,
            )
            if m:
                clean = good_candidate(m.group(1).strip())
                if clean:
                    logger.debug("_parse_recipient QR-ke pattern: %r", clean)
                    return clean
            m = re.search(r"payment\s+to\s+(.+)", line, re.IGNORECASE)
            if m:
                clean = good_candidate(m.group(1).strip())
                if clean:
                    return clean

        # OCR can place ads/cards between a label like "Bayar Ke" and the
        # actual QRIS target. Scan a wider bounded window for that value.
        for i, line in enumerate(lines):
            lower = line.lower()
            if any(kw in lower for kw in _MERCHANT_CONTEXT_KEYWORDS):
                for j in range(i + 1, min(i + 10, len(lines))):
                    cand = lines[j].strip()
                    if any(noise in cand.lower() for noise in (
                        "voucher", "tukar poin", "emas", "paling hemat",
                        "waktu selesai", "berhasil", "detail transaksi",
                    )):
                        continue
                    if _is_label_line(cand):
                        continue
                    clean = good_candidate(cand)
                    if clean:
                        return clean

        for i, line in enumerate(lines):
            lower = line.lower()
            if "pembayaran qr" in lower or "qr bayar" in lower:
                for j in range(i + 1, min(i + 6, len(lines))):
                    cand = re.sub(r"^\s*ke\s+", "", lines[j].strip(), flags=re.IGNORECASE)
                    clean = good_candidate(cand)
                    if clean:
                        return clean

        for line in lines:
            if re.search(r"gobills\s*-\s*pln\s*token|pln\s*token", line, re.IGNORECASE):
                clean = re.sub(r"\s*-\s*\d{6,}.*$", "", line.strip())
                return _normalize_service_merchant(clean)

        # ------------------------------------------------------------------ #
        # Strategy 1: keyword-anchored (logika lama, with relaxed label check #
        # so QRIS-prefixed merchant names are not wrongly blacklisted)       #
        # ------------------------------------------------------------------ #
        for i, line in enumerate(lines):
            if any(kw in line.lower() for kw in _RECIPIENT_KEYWORDS):
                for j in range(i + 1, min(i + 8, len(lines))):
                    cand = lines[j].strip()
                    if not cand:
                        continue
                    if _is_pan_or_account(cand):
                        continue
                    # Relaxed: only reject exact-match or multi-word label lines
                    # (avoids rejecting QRIS-prefixed merchant names due to token "qris")
                    cand_lower = cand.lower()
                    if cand_lower in _LABEL_BLACKLIST:
                        continue
                    if any(
                        lab in cand_lower
                        for lab in _LABEL_BLACKLIST
                        if len(lab.split()) > 1
                    ):
                        continue
                    if any(kw in cand_lower for kw in _RECIPIENT_KEYWORDS):
                        continue
                    if not _has_letters(cand):
                        continue
                    clean = good_candidate(cand)
                    if clean:
                        return clean

        # ------------------------------------------------------------------ #
        # Strategy 2: fallback — collect text-rich candidates, prefer ALL-CAPS#
        # ------------------------------------------------------------------ #
        skip_re = re.compile(
            r"^\s*("
            # Time / pure-numeric
            r"[\d:.,\s]+(wib|wita|wit|am|pm)?"
            # Amount lines
            r"|[\-–—]?\s*(?:rp|idr)\.?\s*[\d.,Oo]+"
            # Network speed / status bar: "0.2KBIs%", "33.3 KB/s"
            r"|[\d.,]+\s*(?:kb/?s|kbis|mb/?s|mbps|%)"
            # Numeric + QR label: "89 QR Bayar"
            r"|\d+\s+(?:qr\s+bayar|pembayaran)"
            # QR / payment UI labels
            r"|qris\s*$|qr\s+bayar|qr\s*code"
            r"|pembayaran(\s+berhasil)?[!.]?"
            r"|transaksi(\s+berhasil)?"
            r"|detail\s+transaksi|transaction\s+details?"
            r"|rincian(?:\s+pembayaran|\s+transaksi)?"
            r"|x|0|@|ok|done"
            r")\s*$",
            re.IGNORECASE,
        )
        # Pattern: "CITY, POSTAL, COUNTRYCODE" — location strings (e.g. "BANDUNG, 40267, ID")
        _location_re = re.compile(
            r"^[A-Z][A-Z\s]+,\s*\d{4,6}\s*,\s*[A-Z]{2}$"  # city, postal, country
            r"|^[A-Z][A-Z\s]+,\s*\d{4,6}$",                 # city, postal
            re.IGNORECASE,
        )

        candidates: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            lower_stripped = stripped.lower()
            if lower_stripped in _BANK_STATUS_NAMES:
                continue
            if lower_stripped in _LABEL_BLACKLIST:
                continue
            if lower_stripped in _CITY_ONLY_SET:
                continue
            if any(status in lower_stripped for status in ("berhasil", "successful", "transaction details", "rincian")):
                continue
            # Normalize OCR digits BEFORE skip_re so "Rp103.ooo" → "Rp103.000" → skipped
            normalized = _normalize_ocr_digits(stripped)
            if skip_re.match(normalized):
                continue
            if _is_pan_or_account(stripped):
                continue
            if _is_label_line(stripped):
                continue
            if _location_re.match(stripped):
                logger.debug("_parse_recipient: skipping location string %r", stripped)
                continue
            if not _has_letters(stripped, min_letters=4):
                continue
            candidates.append(stripped)

        if not candidates:
            return None

        logger.debug("_parse_recipient Strategy2 candidates: %s", candidates[:5])

        # Prefer ALL-CAPS candidates (QRIS merchant names like 'SOP BURTOK CAB ACEH')
        # but exclude location strings even if ALL-CAPS
        upper_cands = [
            c for c in candidates
            if re.search(r"[A-Z]", c) and c == c.upper()
            and not _location_re.match(c)
        ]
        if upper_cands:
            clean = good_candidate(upper_cands[0])
            if clean:
                logger.debug("_parse_recipient: chose ALL-CAPS %r", clean)
                return clean

        return good_candidate(candidates[0])

    def parse(
        self,
        image_path: str | Path,
        return_raw: bool = False,
        pre_ocr_lines: list[str] | None = None,
    ) -> dict[str, Any]:
        """Parse screenshot M-Banking → dict {amount, date, recipient, ...}.

        Args:
            image_path: Path ke gambar.
            return_raw: Jika True, sertakan field 'raw_text' untuk debugging.
        """
        lines = pre_ocr_lines if pre_ocr_lines is not None else self.extract_text_lines(image_path)
        lines = _merge_split_amounts(lines)  # rejoin blocks split by EasyOCR bold font
        full_text = "\n".join(lines)

        recipient   = self._parse_recipient(lines)
        if recipient and _looks_like_bad_merchant(recipient):
            recipient = None
        amount, amount_candidates = self._parse_amount_with_trace(lines)
        sc_category = detect_screenshot_category(lines)

        logger.debug("[MBankingParser] recipient=%r  amount=%s  sc_category=%r",
                     recipient, amount, sc_category)
        logger.debug("[MBankingParser] ocr_lines=%s", lines)

        result: dict[str, Any] = {
            "amount":              amount,
            "date":                self._parse_date_scored(full_text),
            "recipient":           recipient,
            "screenshot_category": sc_category,   # category detected from UI text
            "n_lines":             len(lines),
            "ocr_lines":           lines,
            "amount_candidates":   amount_candidates,
        }
        if return_raw:
            result["raw_text"] = full_text
        return result
