"""extraction_postprocessor.py — Unified field-level postprocessor.

Receives raw OCR lines + route type, returns clean merchant/amount/date
with per-field confidence and a debug trace string.

No model calls here — pure text heuristics.
Designed to work on any receipt or payment screenshot, not just one sample.
"""
from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

_MONTH_MAP: dict[str, str] = {
    "jan": "01", "januari": "01",
    "feb": "02", "februari": "02",
    "mar": "03", "maret": "03",
    "apr": "04", "april": "04",
    "may": "05", "mei": "05",
    "jun": "06", "juni": "06",
    "jul": "07", "juli": "07",
    "aug": "08", "agu": "08", "ags": "08", "agustus": "08",
    "sep": "09", "sept": "09", "september": "09",
    "okt": "10", "oct": "10", "oktober": "10", "october": "10",
    "nov": "11", "november": "11",
    "des": "12", "dec": "12", "desember": "12", "december": "12",
}

# Labels that must never become merchant
_MERCHANT_REJECT: frozenset[str] = frozenset({
    # Category UI labels
    "shopping", "groceries", "grocery", "food", "beverage",
    "food & drink", "food and drink", "food & beverage",
    "transportasi", "transport", "transportation", "travel",
    "makanan & minuman", "makanan dan minuman", "makanan", "minuman",
    "belanja", "belanja & retail", "belanja dan retail", "retail",
    "lainnya", "other", "others",
    "pendidikan", "education",
    "kesehatan", "health", "healthcare",
    "hiburan", "entertainment",
    "tagihan", "bills", "bill", "utilities",
    "category", "kategori", "tipe", "type",
    # Status words
    "success", "berhasil", "selesai", "successful",
    "done", "ok", "completed",
    # Source/fund labels
    "main pocket", "source of fund", "sumber dana",
    "dari rekening", "ke rekening",
    # Technical field labels
    "acquirer name", "acquirer", "terminal id", "reference number",
    "transaction id", "id transaksi", "customer pan", "merchant pan",
    "transaction details", "rincian transaksi", "rincian pembayaran",
    "payment details", "detail transaksi", "informasi transaksi",
    "transfer details", "payment method", "metode pembayaran",
    "from", "to", "date and time", "reference", "referensi",
    "merchant name", "nama merchant", "merchant location",
    # App/bank names that are NOT merchant targets
    "bca", "bank bca", "bank jago", "seabank", "bank mandiri",
    "bni", "bri", "btn", "cimb", "permata", "maybank", "ocbc", "danamon",
    "shopeepay", "gopay", "dana", "ovo", "linkaja", "blu",
    "xendit", "midtrans",
    # Generic noise / payment UI labels
    "free", "qris", "qr", "qris payment successful",
    "pembayaran berhasil", "merchant tidak terdeteksi",
    "qr bayar", "qr code", "scan qr", "bayar qr",
    "pembayaran qr", "pembayaran qris berhasil",
    "rincian pembayaran", "rincian transaksi",
    # Receipt labels
    "subtotal", "sub total", "total", "grand total",
    "tax", "ppn", "pb1", "service", "tunai", "cash",
    "debit", "kredit", "credit", "payment", "pembayaran",
    "change", "kembalian", "kembali", "diskon", "discount",
    "kasir", "cashier", "operator", "receipt", "rcpt",
    "table", "guest", "qty", "item",
})

# City strings alone are not valid merchant names
_CITY_REJECT: frozenset[str] = frozenset({
    "bandung", "jakarta", "surabaya", "medan", "semarang", "makassar",
    "denpasar", "malang", "bogor", "depok", "bekasi", "tangerang",
    "bali", "jogja", "yogyakarta", "solo", "palembang", "balikpapan",
    "banjarmasin", "pontianak", "manado", "pekanbaru", "padang",
    "cimahi", "tasikmalaya", "cilegon", "serang", "jambi",
    "id", "idn", "indonesia",
})

# Lines that start with these prefixes are addresses, not merchant names
_ADDRESS_PREFIXES = (
    "jl.", "jl ", "jalan", "ruko", "perumahan", "komplek",
    "alamat", "telp", "tel.", "phone", "ph.", "no.", "lt.", "lantai",
    "blok", "kel.", "kec.", "kota", "kab.", "(",
)

# Reference/ID line regex — kills date candidates on ID lines
_REF_LINE_RE = re.compile(
    r"\b(transaction\s*id|id\s*transaksi|reference|referensi|no\.?\s*ref|"
    r"customer\s*pan|merchant\s*pan|terminal\s*id|trace|rrn|stan|auth|"
    r"approval|invoice|receipt\s*no|order\s*(sn|no|id)|transaction\s*sn|"
    r"source\s*of\s*fund|account|rekening|phone|telp|token|"
    r"merchant\s*ref\s*id)\b",
    re.IGNORECASE,
)

# Date context keyword → boosts nearby date candidates
_DATE_CONTEXT_KW = frozenset({
    "tanggal", "date", "waktu", "time", "jam",
    "receipt", "rcp", "rcpt", "trx", "transaksi", "transaction",
    "payment", "pembayaran", "selesai", "closed", "open", "printed",
    "tanggal dan waktu", "date and time", "waktu selesai",
    "berhasil", "sukses", "successful",
    "payment date", "transaction date", "transaction time",
})

# Amount keyword priority tiers for receipts
_AMT_TIER1 = frozenset({
    # Explicit due / final-balance
    "amount due", "total due", "balance due", "due",
    # Grand-total family
    "grand total", "final total", "total belanja", "total bayar", "jumlah bayar",
    "total akhir", "jumlah akhir", "total harga", "total pembayaran",
    "total payment", "total paid", "paid amount", "amount paid",
    # Short labels common on printed receipts
    "total", "jumlah", "tagihan", "amount", "amt",
    "bayar",
    # Abbreviated totals used on narrow thermal receipts
    "tl", "ttl",
    # Net / nett (common in restaurant / hospitality receipts)
    "net", "nett", "net amount", "net total", "nett amount",
})
_AMT_TIER2 = frozenset({
    # Payment-method / tender lines — win only if no tier-1 exists
    "tunai", "cash", "debit", "credit", "qris", "payment",
    "paid", "lunas", "tender", "mastercard", "visa", "card",
})
_AMT_TIER3 = frozenset({
    # Adjustment / component rows — penalised heavily when tier-1 exists
    "subtotal", "sub total", "service charge", "service", "serv", "svc",
    "tax", "ppn", "pb1", "sc", "vat", "fee",
    "discount", "disc", "diskon", "promo", "voucher",
    "change", "kembalian", "kembali",
})

# Amount keywords to reject for m-banking (screen noise)
_MBANKING_AMT_REJECT_KW = frozenset({
    "transaction id", "id transaksi", "reference", "referensi",
    "no. ref", "merchant pan", "customer pan", "terminal id",
    "token", "source of fund", "sumber dana", "rekening", "account",
    "phone", "telepon", "tanggal", "date", "time", "waktu",
    "rrn", "stan", "trace", "auth", "approval", "invoice",
    "order sn", "transaction sn", "merchant ref",
})

_RECEIPT_AMOUNT_REJECT_RE = re.compile(
    r"\b(http|https|www\.|\.com|\.co\.id|link|url|instagram|wifi|password|"
    r"coupon|kupon|voucher|card|kartu|phone|telp|telepon|no\s*hp|hp|"
    r"cashier|kasir|operator|receipt\s*no|rcpt|invoice|ref|reference|"
    r"order\s*(?:id|no)|id\s*transaksi|no\.?\s*struk)\b",
    re.IGNORECASE,
)

_RECEIPT_ITEM_WORD_RE = re.compile(
    r"\b(nasi|ayam|mie|bakso|kopi|coffee|tea|water|roti|pasta|burger|"
    r"rice|noodle|drink)\b",
    re.IGNORECASE,
)

_GENERIC_SINGLE_WORD_ITEM_RE = re.compile(
    r"^(?:nasi|ayam|mie|bakso|kopi|coffee|tea|water|roti|pasta|"
    r"burger|rice|noodle|drink)$",
    re.IGNORECASE,
)

_RECEIPT_MERCHANT_CUE_RE = re.compile(
    r"\b(warung|toko|shop|store|mart|resto|restaurant|cafe|coffee|kopi|"
    r"bakery|dapur|kedai|rumah\s+makan|rm\b|pt\b|cv\b)\b",
    re.IGNORECASE,
)

_RECEIPT_FOOTER_RE = re.compile(
    r"\b(thank\s*you|terima\s+kasih|instagram|wifi|password|kritik|saran|"
    r"link|coupon|kupon|voucher|card|cashier|kasir|operator|lun[a4]s)\b",
    re.IGNORECASE,
)


# ============================================================
# OCR NORMALIZATION HELPERS
# ============================================================

def _fix_ocr_digits(text: str) -> str:
    """Fix common OCR digit confusions in numeric context."""
    # O/o → 0 in numeric context
    text = re.sub(r"(?<=\d)[Oo]", "0", text)
    text = re.sub(r"[Oo](?=\d)", "0", text)
    text = re.sub(r"([.,])([Oo]+)", lambda m: m.group(1) + "0" * len(m.group(2)), text)
    # I/l → 1 between digits or after letter
    text = re.sub(r"(?<=\d)[Il](?=\d)", "1", text)
    text = re.sub(r"(?<=[a-zA-Z])([Il])(?=\d)", "1", text)
    # S → 5 in numeric context
    text = re.sub(r"(?<=\d)S(?=\d)", "5", text)
    text = re.sub(r"(?<=\d)S(?=[.,]\d)", "5", text)
    # B → 8 in numeric context
    text = re.sub(r"(?<=\d)B(?=[\d.,])", "8", text)
    text = re.sub(r"(?<=\d)b(?=[\d.,])", "8", text)
    return text


def _fix_date_ocr(text: str) -> str:
    """Normalize OCR quirks specifically around date fragments."""
    text = _fix_ocr_digits(text or "")
    # Collapse spaces around date separators: "04 / 01 / 2026" → "04/01/2026"
    text = re.sub(r"(?<=\d)\s*([/.\-_|])\s*(?=\d)", r"\1", text)
    # Pipe misread as 1: "2024|07|18" — keep as-is, separator normalizer handles it
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# DATE PARSER
# ============================================================

_DATE_PATTERNS = [
    # "11 Apr 2026", "4 January 2026", "13 Agustus 2025" — named month (highest priority)
    re.compile(
        r"(\d{1,2})\s+([A-Za-z]{3,}?)\.?\s+(\d{2,4})",
        re.IGNORECASE,
    ),
    # "April 9, 2026" / "Apr 9 2026" — month-first English
    re.compile(
        r"([A-Za-z]{3,}?)\.?\s+(\d{1,2}),?\s+(\d{2,4})",
        re.IGNORECASE,
    ),
    # ISO "2026-04-11" / "2026/04/11" / "2026.04.11"
    re.compile(r"\b(\d{4})[/.\-_|](\d{1,2})[/.\-_|](\d{1,2})\b"),
    # "11/04/2026" / "11-04-26" (slash or dash — handled before dot pattern)
    re.compile(r"\b(\d{1,2})[/\-_|](\d{1,2})[/\-_|](\d{2,4})\b"),
    # Dot separator: "11.04.2026" / "18.7.2024" — must be explicit to avoid
    # colliding with decimal numbers.  Allows single-digit month/day (18.7.2024).
    re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b"),
    # Compact "DDMMYYYY" — rare but present on some thermal receipts (no separator)
    re.compile(r"\b(\d{2})(\d{2})(\d{4})\b"),
]

# Label prefixes to strip before date parsing
_DATE_LABEL_RE = re.compile(
    r"(?i)^(?:tanggal\s+dan\s+waktu|date\s+and\s+time|waktu\s+selesai|"
    r"payment\s+date|transaction\s+date|transaction\s+time|"
    r"tanggal|date|waktu|time|jam|waktu\s+pembayaran)\s*[:\-]?\s*",
)


def _expand_year(y: str) -> int:
    if len(y) == 2:
        n = int(y)
        return 2000 + n if n <= 35 else 1900 + n
    return int(y)


def _valid_date(d: int, m: int, y: int) -> bool:
    if not (1 <= d <= 31 and 1 <= m <= 12 and 2010 <= y <= 2035):
        return False
    try:
        datetime(y, m, d)
        return True
    except ValueError:
        return False


def _fmt_date(d: int, m: int, y: int) -> str:
    return f"{d:02d}/{m:02d}/{y:04d}"


def _try_parse_date(raw: str) -> str:
    """Try all patterns on a single raw string, return DD/MM/YYYY or ''."""
    s = _fix_date_ocr(raw)
    # Strip trailing time/timezone variants:
    #   "11 Apr 2026 - 19:03:20 WIB" → "11 Apr 2026"
    #   "18.7.2024 08:49"            → "18.7.2024"
    #   "4 January 2026, 17.59"      → "4 January 2026"
    s = re.sub(r"\s*[-–]\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*\w{2,4})?\s*$", "", s).strip()
    s = re.sub(r",\s*\d{1,2}[:.]\d{2}(?:[:.]\d{2})?\s*(?:\w{2,4})?\s*$", "", s).strip()
    s = re.sub(r"\s+\d{1,2}[:.]\d{2}(?::\d{2})?\s*(?:wib|wita|wit|am|pm)?\s*$", "", s, flags=re.IGNORECASE).strip()
    # Dot-separated time without colon: "4 January 2026, 17.59"  (already handled above)
    # Space-separated time: "18.7.2024 08:49"
    s = re.sub(r"\s+\d{2}[:.]\d{2}(?:[:.]\d{2})?\s*$", "", s).strip()

    # Pattern 1: day + named-month + year
    m1 = re.match(r"(\d{1,2})\s+([A-Za-z]{3,})\.?\s+(\d{2,4})", s, re.IGNORECASE)
    if m1:
        mon_key = m1.group(2).lower()[:9]  # up to 9 chars for "september"
        # Try progressively shorter prefix
        for ln in (len(mon_key), min(len(mon_key), 4), 3):
            mn = _MONTH_MAP.get(mon_key[:ln])
            if mn:
                break
        if mn:
            d_, y_ = int(m1.group(1)), _expand_year(m1.group(3))
            m_ = int(mn)
            if _valid_date(d_, m_, y_):
                return _fmt_date(d_, m_, y_)

    # Pattern 2: named-month + day + year (English style)
    m2 = re.match(r"([A-Za-z]{3,})\.?\s+(\d{1,2}),?\s+(\d{2,4})", s, re.IGNORECASE)
    if m2:
        mon_key = m2.group(1).lower()[:9]
        for ln in (len(mon_key), min(len(mon_key), 4), 3):
            mn = _MONTH_MAP.get(mon_key[:ln])
            if mn:
                break
        if mn:
            d_, y_ = int(m2.group(2)), _expand_year(m2.group(3))
            m_ = int(mn)
            if _valid_date(d_, m_, y_):
                return _fmt_date(d_, m_, y_)

    # Pattern 3: ISO YYYY-MM-DD
    m3 = re.match(r"(\d{4})[/.\-_|](\d{1,2})[/.\-_|](\d{1,2})", s)
    if m3:
        y_, m_, d_ = _expand_year(m3.group(1)), int(m3.group(2)), int(m3.group(3))
        if _valid_date(d_, m_, y_):
            return _fmt_date(d_, m_, y_)

    # Pattern 4: DD/MM/YYYY and variants (including dot: 18.7.2024)
    m4 = re.match(r"(\d{1,2})[/.\-_|](\d{1,2})[/.\-_|](\d{2,4})", s)
    if m4:
        d_, m_, y_ = int(m4.group(1)), int(m4.group(2)), _expand_year(m4.group(3))
        # Swap if month > 12 and day <= 12 (MM/DD/YYYY foreign format)
        if m_ > 12 and 1 <= d_ <= 12:
            d_, m_ = m_, d_
        if _valid_date(d_, m_, y_):
            return _fmt_date(d_, m_, y_)

    # Pattern 5: compact DDMMYYYY (no separator)
    m5 = re.match(r"^(\d{2})(\d{2})(\d{4})$", s)
    if m5:
        d_, m_, y_ = int(m5.group(1)), int(m5.group(2)), int(m5.group(3))
        if _valid_date(d_, m_, y_):
            return _fmt_date(d_, m_, y_)

    return ""


def extract_date(
    ocr_lines: list[str],
    route_type: str = "receipt",
) -> tuple[str, float, str]:
    """
    Extract best date from OCR lines.

    Returns:
        (date_str_DD/MM/YYYY, confidence, debug_reason)
    """
    trace_parts: list[str] = []
    candidates: list[dict[str, Any]] = []

    norm_lines = [_fix_date_ocr(line) for line in ocr_lines]

    for i, (raw_line, norm_line) in enumerate(zip(ocr_lines, norm_lines)):
        lower_line = norm_line.lower()
        # Context: look at window of +/-2 lines so split label/value OCR still works.
        window = " ".join(norm_lines[max(0, i - 2): min(len(norm_lines), i + 3)]).lower()

        # Skip lines that are clearly ID/reference lines
        if _REF_LINE_RE.search(lower_line) and not any(k in lower_line for k in ("tanggal", "date", "waktu", "time")):
            continue

        # Strip label prefix from line before searching for date
        stripped_line = _DATE_LABEL_RE.sub("", norm_line).strip()

        # Try the stripped line
        for search_text in [stripped_line, norm_line]:
            # Use all patterns to find date fragments
            for pat in _DATE_PATTERNS:
                for match in pat.finditer(search_text):
                    raw_match = match.group(0)
                    parsed = _try_parse_date(raw_match)
                    if not parsed:
                        continue
                    # Score this candidate
                    score = 50
                    reason_parts = ["found"]
                    if any(k in lower_line for k in ("tanggal", "date", "waktu", "time", "selesai", "berhasil")):
                        score += 50
                        reason_parts.append("date_label_line")
                    elif any(k in window for k in ("tanggal", "date", "waktu", "time", "selesai", "berhasil", "transaksi", "transaction", "payment", "pembayaran")):
                        score += 35
                        reason_parts.append("date_context_window")
                    # Named-month dates are more reliable
                    if re.search(r"[A-Za-z]{3,}", raw_match):
                        score += 20
                        reason_parts.append("named_month")
                    # Upper area bonus
                    if i <= max(3, len(norm_lines) // 4):
                        score += 15
                        reason_parts.append("upper_area")
                    elif i <= max(6, len(norm_lines) // 2):
                        score += 8
                        reason_parts.append("mid_area")
                    # Near time string
                    if re.search(r"\b\d{1,2}[:.]\d{2}(?::\d{2})?\b", norm_line):
                        score += 8
                        reason_parts.append("near_time")
                    # Penalise if looks like a reference ID context
                    if re.search(r"[A-Z0-9]{5,}-[A-Z0-9-]{4,}", norm_line, re.I):
                        score -= 60
                        reason_parts.append("id_like_penalty")
                    candidates.append({
                        "raw": raw_match,
                        "parsed": parsed,
                        "line": i,
                        "score": score,
                        "reason": "|".join(reason_parts),
                        "line_text": norm_line,
                    })

    # Deduplicate by (parsed_value, line_index)
    dedup: dict[tuple[str, int], dict[str, Any]] = {}
    for c in candidates:
        key = (c["parsed"], c["line"])
        if key not in dedup or c["score"] > dedup[key]["score"]:
            dedup[key] = c

    sorted_cands = sorted(dedup.values(), key=lambda c: c["score"], reverse=True)

    for c in sorted_cands[:8]:
        trace_parts.append(
            f"  line[{c['line']}] raw={c['raw']!r} → {c['parsed']} "
            f"score={c['score']} reason={c['reason']}"
        )

    if not sorted_cands:
        return "", 0.0, "no_date_found\n" + "\n".join(trace_parts)

    best = sorted_cands[0]
    conf = 0.92 if re.search(r"[A-Za-z]{3,}", best["raw"]) else (0.80 if best["score"] >= 90 else 0.65)
    return best["parsed"], conf, f"selected_score={best['score']} reason={best['reason']}\n" + "\n".join(trace_parts)


# ============================================================
# AMOUNT PARSER
# ============================================================

def _amount_doc_context(context: str = "", doc_type: str = "") -> bool:
    ctx = f"{context or ''}\n{doc_type or ''}".lower()
    return any(k in ctx for k in ("compact_qr_card", "qr bayar", "pembayaran qr", "pembayaran qris"))


def _parse_money_token(
    raw: str,
    idr_context: bool = False,
    *,
    context: str = "",
    doc_type: str = "",
) -> float | None:
    """Parse a single money token to float. Handles IDR thousands and foreign decimals."""
    s = _fix_ocr_digits(str(raw))
    s = re.sub(r"(?i)\b(rp\.?|idr|cad|usd|aud|sgd|eur)\b", "", s)
    s = s.replace("$", "").replace("€", "").replace("£", "")
    s = re.sub(r"[\-–—+]", "", s)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"^[^\d]+|[^\d]+$", "", s)
    if not s or not re.search(r"\d", s):
        return None
    compact_qr_context = _amount_doc_context(context, doc_type)

    def digits_only(t: str) -> float | None:
        d = re.sub(r"\D", "", t)
        if not d:
            return None
        v = float(d)
        return v if 1 <= v <= 1_000_000_000 else None

    if "," in s and "." in s:
        last_dot = s.rfind(".")
        last_comma = s.rfind(",")
        decimal_sep = "." if last_dot > last_comma else ","
        frac = s[max(last_dot, last_comma) + 1:]
        if len(frac) in {1, 2}:
            # cents-style decimal
            whole = re.sub(r"\D", "", s[:max(last_dot, last_comma)])
            if not whole:
                return None
            if idr_context:
                return float(whole) if 1 <= float(whole) <= 1_000_000_000 else None
            try:
                v = float(f"{whole}.{frac}")
                return v if 1 <= v <= 1_000_000_000 else None
            except ValueError:
                return None
        return digits_only(s)

    sep = "." if "." in s else "," if "," in s else ""
    if sep:
        parts = s.split(sep)
        if (
            compact_qr_context
            and idr_context
            and len(parts) == 2
            and len(parts[0]) == 3
            and len(parts[1]) == 3
            and parts[1].startswith("00")
            and parts[1] != "000"
        ):
            return digits_only(parts[0] + parts[1][:2])
        if len(parts) > 2 and all(len(p) == 3 for p in parts[1:] if p):
            # "1.591.600" or "1,591,600" → thousands
            return digits_only(s)
        if len(parts) == 2:
            left, right = parts
            if (
                compact_qr_context
                and idr_context
                and len(left) == 3
                and len(right) == 3
                and right.startswith("00")
                and right != "000"
            ):
                return digits_only(left + right[:2])
            if len(right) == 3 and 1 <= len(left) <= 3:
                # "38.000" → 38000
                return digits_only(s)
            # OCR artifact: >3 digits after separator (e.g. "10.00000", "94.80009").
            # In IDR context the first 3 digits form the thousands group;
            # excess digits are OCR noise — trim them.
            if len(right) > 3 and idr_context:
                return digits_only(left + right[:3])
            if len(right) in {1, 2}:
                # decimal: "46.15" or "38,50"
                if idr_context:
                    # treat as thousands-truncated, return just whole part
                    return digits_only(left)
                try:
                    left_d = re.sub(r"\D", "", left)
                    v = float(f"{left_d}.{right}")
                    return v if 1 <= v <= 1_000_000_000 else None
                except ValueError:
                    return None
        return digits_only(s)

    return digits_only(s)


def _money_debug_normalized(raw: str, *, idr_context: bool, context: str = "", doc_type: str = "") -> str:
    parsed = _parse_money_token(raw, idr_context=idr_context, context=context, doc_type=doc_type)
    if parsed is None:
        return ""
    return str(int(parsed)) if float(parsed).is_integer() else str(parsed)


def _trim_merchant_location(text: str) -> str:
    """Remove address/location tails without cutting normal multi-word names."""
    clean = re.sub(r"\s+", " ", text or "").strip(" ,.;:-")
    if not clean:
        return clean
    # Cut only when an address marker begins a new segment. This keeps normal
    # branch names such as "The Harvest Cakes, Daan Mogot" but removes the
    # street/kecamatan tail after it.
    clean = re.split(
        r"\s*(?:;|\||/|\s-\s)\s*(?=(?:jl\.?|ji\.?|jalan|ruko|komplek|perumahan|blok|rt\.?|rw\.?|"
        r"kel\.?|kec\.?|kecamatan|kelurahan|kota|kab\.?|kabupaten|kode\s+pos|no\.?|lt\.?|lantai)\b)",
        clean,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    clean = re.sub(
        r"\s*,\s*(?:jl\.?|ji\.?|jalan|ruko|komplek|perumahan|blok|rt\.?|rw\.?|kel\.?|kec\.?|"
        r"kecamatan|kelurahan|kota|kab\.?|kabupaten|kode\s+pos|no\.?\s+\d|lt\.?|lantai)\b.*$",
        "",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\s+\b(?:jl\.?|ji\.?|jalan|rt\.?|rw\.?|kec\.?|kel\.?|kecamatan|kelurahan|kode\s+pos)\s+[A-Za-z0-9].*$",
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


def _sanitize_ocr_amount(text: str) -> str:
    """Remove OCR artifacts from amount strings before parsing.

    EasyOCR sometimes appends repeated zeros or noise digits after a standard
    IDR thousands group when reading compact QR cards or bold display fonts:
      "IDR 10.00000"  → "IDR 10.000"   (repeated-zero artifact)
      "IDR 26.00009"  → "IDR 26.000"   (noise digit after zeros)
      "Rp 94.80009"   → "Rp 94.800"    (noise digit after valid group)

    Rule: for any  N.DDDD+ pattern, keep at most 3 post-dot digits.
    Multi-dot chains (1.000.000) are left untouched because each group
    already has exactly 3 digits.
    """
    def _trim_dot_group(m: re.Match) -> str:
        left, right = m.group(1), m.group(2)
        # Only trim the rightmost group when it has 4+ digits
        return f"{left}.{right[:3]}"

    # Match a digit sequence, a dot, then 4+ digits (no more dots after)
    return re.sub(r"(\d+)\.(\d{4,})(?!\d*\.)", _trim_dot_group, text)


# ── Document-type classifier (lightweight, keyword/heuristic only) ────────────

_COMPACT_QR_KWS = frozenset({
    "qr bayar", "pembayaran qr", "pembayaran qris",
})
_DETAIL_KWS = frozenset({
    "transaction details", "merchant name", "date and time",
    "payment method", "source of fund", "terminal id",
    "rincian transaksi", "metode pembayaran", "nama merchant",
    "customer pan", "merchant pan", "reference number",
})
_EWALLET_KWS = frozenset({
    "gopay", "shopeepay", "ovo", "dana", "linkaja",
    "pembayaran berhasil", "bayar ke", "transaksi berhasil",
    "berhasil dikirim",
})
_RECEIPT_KWS = frozenset({
    "subtotal", "kasir", "cashier", "kembalian", "kembali",
    "tunai", "struk", "nota", "ppn", "qty", "pcs", "receipt",
    "guest folio", "folio", "balance due", "charges", "credits",
    "room", "receipt no",
})


def classify_document_type(ocr_lines: list[str]) -> str:
    """Classify document type from OCR lines using lightweight heuristics.

    Returns one of:
      "compact_qr_card"          – short QR-payment crop
      "mbanking_transaction_detail" – structured detail page with many fields
      "ewallet_receipt_screen"   – GoPay/ShopeePay/OVO success screen
      "photo_receipt"            – physical receipt photograph
      "other_unknown"            – none of the above
    """
    full = "\n".join(ocr_lines).lower()
    n = len(ocr_lines)
    receipt_brand_hit = bool(re.search(
        r"\b(alfamart|alfamidi|indomaret|minimarket|supermarket)\b",
        full,
    ))
    receipt_payment_hit = bool(re.search(
        r"\b(total|grand\s+total|subtotal|tunai|kembali|kembalian|kasir|cashier|ppn|terima\s+kasih)\b",
        full,
    ))
    strong_payment_infra = bool(re.search(
        r"\b(merchant\s+pan|customer\s+pan|terminal\s+id|reference\s+number|transaction\s+id|"
        r"source\s+of\s+fund|acquirer\s+name)\b",
        full,
    ))
    qr_payment_hint = bool(re.search(r"\b(qr\s+bayar|pembayaran\s+qris|pembayaran\s+qr)\b", full))
    if receipt_brand_hit and receipt_payment_hit and not strong_payment_infra:
        return "photo_receipt"

    # Compact QR card: ≤ 15 lines, has "QR Bayar" / "Pembayaran QR"
    has_qr_label = any(kw in full for kw in _COMPACT_QR_KWS)
    has_currency = bool(re.search(r"\b(?:rp|idr)\s*[\d.,oOIlSBb]{4,}", full))
    has_ke_merchant = bool(re.search(r"(?:^|\n|\b)ke\s+[a-z0-9]", full))
    if has_qr_label and (n <= 20 or (has_currency and has_ke_merchant)):
        return "compact_qr_card"

    detail_score = sum(1 for kw in _DETAIL_KWS if kw in full)
    ewallet_score = sum(1 for kw in _EWALLET_KWS if kw in full)
    receipt_score = sum(1 for kw in _RECEIPT_KWS if kw in full)
    receipt_score += min(6, len(re.findall(
        r"\b(subtotal|sub total|grand total|total due|amount due|balance due|"
        r"nett?|tax|vat|service|discount|cash|tunai|change|kembali|kembalian)\b",
        full,
    )))
    item_like = 0
    for line in ocr_lines:
        lower = _fix_ocr_digits(line).lower()
        if re.search(r"\b\d+\s*x\b|\b\d+x\b|\bx\d+\b|\bqty\b|\bpcs\b|@", lower):
            item_like += 1
        elif re.search(r"[a-z]{3,}.*(?:\d{1,3}[.,]\d{3}|\d+[.,]\d{2})\s*$", lower):
            item_like += 1
    receipt_score += min(6, item_like)
    if receipt_payment_hit and item_like >= 2:
        receipt_score += 4
    if receipt_brand_hit and receipt_payment_hit:
        receipt_score += 5

    screenshot_score = detail_score * 2 + ewallet_score * 2
    if re.search(r"\b(customer pan|merchant pan|terminal id|reference|rrn|stan|source of fund)\b", full):
        screenshot_score += 4
    if qr_payment_hint and (detail_score > 0 or ewallet_score > 0 or n <= 20):
        screenshot_score += 3
    if re.search(r"\b(share|bagikan|category|kategori|main pocket|saldo)\b", full):
        screenshot_score += 2

    # Printed receipts may include QRIS as a payment method; do not let QR words
    # alone override item-list and total/cash evidence.
    if qr_payment_hint and receipt_score >= screenshot_score and not strong_payment_infra:
        screenshot_score = max(0, screenshot_score - 2)

    if screenshot_score >= 4 and screenshot_score - receipt_score >= 2:
        return "mbanking_transaction_detail" if detail_score >= ewallet_score else "ewallet_receipt_screen"
    if receipt_score >= 4 and receipt_score - screenshot_score >= 2:
        return "photo_receipt"

    return "other_unknown"


_FOREIGN_CURRENCY_RE = re.compile(
    r"\b(cad|usd|aud|sgd|eur|gbp)\b|[$€£]", re.IGNORECASE
)
_IDR_CURRENCY_RE = re.compile(r"\b(rp|idr)\b", re.IGNORECASE)


def detect_receipt_currency(ocr_lines: list[str]) -> str:
    """Detect the primary currency from OCR lines.

    Returns an ISO code ("IDR", "CAD", "USD", "EUR", "AUD", "SGD", "GBP")
    or "FOREIGN" if a foreign symbol is present but the specific code is unknown.
    Defaults to "IDR" (Indonesian context) when nothing distinctive is found.
    """
    full = " ".join(ocr_lines)
    # Explicit IDR marker wins immediately
    if _IDR_CURRENCY_RE.search(full):
        return "IDR"
    lower = full.lower()
    if re.search(r"\bcad\b", lower):
        return "CAD"
    if re.search(r"\busd\b", lower):
        return "USD"
    if re.search(r"\beur\b|€", lower):
        return "EUR"
    if re.search(r"\baud\b", lower):
        return "AUD"
    if re.search(r"\bsgd\b", lower):
        return "SGD"
    if re.search(r"\bgbp\b|£", lower):
        return "GBP"
    if "$" in full:
        return "USD"  # bare $ → assume USD
    return "IDR"


def _extract_amounts_from_line(
    line: str,
    idr_context: bool = True,
    *,
    context: str = "",
    doc_type: str = "",
) -> list[float]:
    """Extract all plausible money values from a single line.

    For lines with explicit foreign currency prefix (CAD/USD/EUR/…) and no
    Rp/IDR prefix, decimal is treated as fractional (e.g. CAD$46.15 → 46.15)
    rather than a thousand-truncated IDR amount.
    """
    line = _fix_ocr_digits(_sanitize_ocr_amount(line))
    values: list[float] = []
    has_idr = bool(_IDR_CURRENCY_RE.search(line))
    money_re = re.compile(
        r"(?:\b(?:rp|idr|cad|usd|aud|sgd|eur|gbp)\b\.?\s*)?[$€£]?\s*"
        r"(\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?|\d+[.,]\d{1,2}|\d{4,})",
        re.IGNORECASE,
    )
    for m in money_re.finditer(line):
        raw = m.group(0)
        # Per-match currency detection: foreign prefix → disable IDR context so
        # "CAD$46.15" → 46.15 instead of 46 or 4615.
        match_has_foreign = bool(_FOREIGN_CURRENCY_RE.search(raw))
        match_has_idr = bool(_IDR_CURRENCY_RE.search(raw))
        use_idr = (idr_context or has_idr) and not (match_has_foreign and not match_has_idr)
        v = _parse_money_token(raw, idr_context=use_idr, context=context, doc_type=doc_type)
        if v is not None and 1 <= v <= 1_000_000_000:
            values.append(v)
    return values


def _kw_tier(lower: str) -> int:
    """Return keyword tier: 1=highest, 2=medium, 3=low, 0=none."""
    if re.search(
        r"\b(subtotal|sub\s+total|service\s+charge|service|serv|svc|tax|ppn|pb1|"
        r"vat|fee|discount|disc|diskon|promo|voucher|change|kembalian|kembali)\b",
        lower,
    ):
        return 3
    for kw in _AMT_TIER1:
        if " " in kw:
            if kw in lower:
                return 1
        else:
            if re.search(rf"\b{re.escape(kw)}\b", lower):
                return 1
    for kw in _AMT_TIER2:
        if re.search(rf"\b{re.escape(kw)}\b", lower):
            return 2
    for kw in _AMT_TIER3:
        if " " in kw:
            if kw in lower:
                return 3
        else:
            if re.search(rf"\b{re.escape(kw)}\b", lower):
                return 3
    return 0


def extract_amount_receipt(
    ocr_lines: list[str],
) -> tuple[float, float, str]:
    """
    Extract final payable amount from receipt OCR lines.

    Priority: DUE > TOTAL > PAYMENT > nothing (SUBTOTAL/TAX rejected when TOTAL exists).

    Returns:
        (amount, confidence, debug_trace)
    """
    norm = [_fix_ocr_digits(ln) for ln in ocr_lines]
    n = max(len(norm), 1)

    candidates: list[dict[str, Any]] = []

    # Detect whether any tier-1 total keyword exists
    total_kw_exists = any(_kw_tier(ln.lower()) == 1 for ln in norm)
    # Detect DUE specifically
    due_kw_exists = any(
        re.search(r"\b(amount\s+due|total\s+due|balance\s+due|due)\b", ln.lower())
        for ln in norm
    )

    def add_candidate(idx: int, value: float, base_score: int, reason: str) -> None:
        lower = norm[idx].lower()
        prev_context = " ".join(norm[max(0, idx - 2):idx]).lower()
        next_context = " ".join(norm[idx + 1:min(n, idx + 3)]).lower()
        window_context = f"{prev_context} {lower} {next_context}"
        tier = _kw_tier(lower)
        score = base_score
        has_final_kw = bool(re.search(
            r"\b(grand\s+total|final\s+total|total\s+bayar|total\s+due|amount\s+due|"
            r"balance\s+due|due|total|bayar|tunai|jumlah)\b",
            lower,
        ))
        has_due_kw = bool(re.search(r"\b(amount\s+due|total\s+due|balance\s+due|due)\b", lower))
        has_total_kw = bool(re.search(
            r"\b(grand\s+total|final\s+total|total\s+belanja|total\s+bayar|jumlah\s+bayar|"
            r"total\s+payment|total\s+paid|paid\s+amount|total\s+harga|"
            r"net\s+amount|net\s+total|nett\s+amount|total|tl|ttl|net|nett)\b",
            lower,
        ))
        payment_only_kw = bool(re.search(r"\b(tunai|cash|debit|credit|qris|payment|paid|lunas|tender|bayar)\b", lower))
        payment_is_final_total = bool(re.search(r"\b(total\s+bayar|jumlah\s+bayar|total\s+payment|total\s+paid|paid\s+amount)\b", lower))

        # DUE keywords: highest priority
        if has_due_kw:
            score += 200
            reason += "|due_kw"

        # TOTAL keyword family (keep in sync with _AMT_TIER1)
        if has_total_kw:
            score += 120
            reason += "|total_kw"

        # AMOUNT / JUMLAH keywords
        if re.search(r"\b(amount|jumlah|tagihan|nominal)\b", lower):
            score += 80
            reason += "|amount_kw"

        # Payment method (medium tier — only wins when no total found)
        if tier == 2 and not total_kw_exists:
            score += 50
            reason += "|payment_kw_no_total"
        elif tier == 2:
            score += 15
            reason += "|payment_kw"
        if total_kw_exists and payment_only_kw and not (has_total_kw or has_due_kw or payment_is_final_total):
            score -= 180
            reason += "|payment_line_penalty_total_exists"

        # Tier-3 (subtotal, tax, service, change) — penalise heavily
        if tier == 3:
            penalty = -150 if total_kw_exists else -60
            score += penalty
            reason += f"|tier3_penalty({penalty})"
        # Change / kembalian — always reject
        if re.search(r"\b(change|kembalian|kembali)\b", lower):
            score -= 500
            reason += "|change_reject"
        if _RECEIPT_AMOUNT_REJECT_RE.search(lower):
            if has_final_kw:
                score -= 120
                reason += "|receipt_noise_soft_penalty"
            else:
                score -= 900
                reason += "|receipt_noise_reject"
        if _RECEIPT_AMOUNT_REJECT_RE.search(prev_context) and not re.search(
            r"\b(grand\s+total|total|amount\s+due|total\s+due|balance\s+due|due|bayar|tunai)\b",
            lower,
        ):
            score -= 450
            reason += "|prev_noise_label_reject"
        if re.search(r"(?:https?://|www\.|\.com|\.co\.id|link|url|/f/|/[a-z0-9]{4,})", lower):
            score -= 900
            reason += "|url_reject"
        if re.search(r"(?:https?://|www\.|\.com|\.co\.id|link|url|/f/|/[a-z0-9]{4,})", prev_context):
            score -= 650
            reason += "|prev_url_reject"
        digits_only = re.sub(r"\D", "", norm[idx])
        if len(digits_only) > 8 and not has_final_kw:
            score -= 650
            reason += "|long_noise_reject"
        if len(digits_only) > 10:
            score -= 700 if not has_final_kw else 250
            reason += "|very_long_digit_penalty"
        # Cash/tunai with change → do not choose cash as final amount
        if re.search(r"\b(tunai|cash)\b", lower) and any(
            re.search(r"\b(change|kembalian|kembali)\b", norm[j].lower())
            for j in range(max(0, idx - 3), min(n, idx + 4))
        ):
            score -= 200
            reason += "|cash_with_change_reject"

        # Look-back: if the 1-2 preceding lines contain change/cash keyword,
        # treat this value as a change/cash amount (keyword on own line, value on next line)
        _prev_flagged = False
        for _prev_idx in range(max(0, idx - 2), idx):
            _prev_lower = norm[_prev_idx].lower()
            if re.search(r"\b(change|kembalian|kembali)\b", _prev_lower):
                score -= 500
                reason += "|change_reject"
                _prev_flagged = True
                break
            if (not _prev_flagged
                    and re.search(r"\b(tunai|cash)\b", _prev_lower)
                    and any(
                        re.search(r"\b(change|kembalian|kembali)\b", norm[_j].lower())
                        for _j in range(max(0, idx - 4), min(n, idx + 2))
                    )):
                score -= 200
                reason += "|cash_with_change_reject"
                _prev_flagged = True
                break

        # Reference / ID lines
        if _REF_LINE_RE.search(lower):
            score -= 120
            reason += "|ref_id_penalty"

        # Item row patterns: "1 x 58000", "1 ITEM NAME price"
        if re.search(r"\b\d+\s*x\b|\b\d+x\b|\bqty\b|\bpcs\b|@", lower):
            score -= 150
            reason += "|qty_row_reject"
        if re.match(r"^\s*\d+\s+[A-Za-z]", norm[idx]) and tier == 0:
            score -= 120
            reason += "|item_row_reject"

        # Lines with alphabetic text but NO total/amount keyword → likely item name line
        if (re.search(r"[A-Za-z]{3,}", norm[idx])
                and tier == 0
                and not re.search(r"\b(rp|idr)\b", lower, re.IGNORECASE)):
            score -= 70
            reason += "|plain_text_no_kw_penalty"

        # Date lines
        if re.search(r"\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}\b", norm[idx]):
            score -= 100
            reason += "|date_penalty"

        if value < 1000 and not has_final_kw:
            score -= 250
            reason += "|too_small_without_total"

        # Bottom-of-receipt bonus (totals usually at bottom 40%)
        if idx >= int(n * 0.60):
            score += 25
            reason += "|bottom_section"

        candidates.append({
            "line": idx,
            "text": ocr_lines[idx],
            "value": value,
            "score": score,
            "reason": reason,
        })

    # Pass 1: same-line amounts
    for i, line in enumerate(norm):
        vals = _extract_amounts_from_line(line, idr_context=True)
        for v in vals:
            add_candidate(i, v, 30, "same_line")

    # Pass 2: keyword-on-own-line → look ahead for amount.
    # Only tier-1 (TOTAL/DUE/NET/TL/…) triggers lookahead.
    # Tier-2 (CASH/DEBIT) and tier-3 (CHANGE/TAX) must NOT — their values
    # are handled by same-line scoring and look-back rejection instead.
    for i, line in enumerate(norm):
        lower = line.lower()
        if _kw_tier(lower) == 1 and not _extract_amounts_from_line(line, idr_context=True):
            for j in range(i + 1, min(i + 5, n)):
                vals = _extract_amounts_from_line(norm[j], idr_context=True)
                for v in vals:
                    add_candidate(j, v, 60, f"next_line_kw[{i}]")
                if vals:
                    break

    # Fallback: if no candidates at all, use bottom lines
    if not candidates:
        bottom = max(0, n - max(3, n * 4 // 10))
        for i in range(bottom, n):
            for v in _extract_amounts_from_line(norm[i], idr_context=True):
                add_candidate(i, v, 20, "bottom_fallback")

    if not candidates:
        return 0.0, 0.0, "no_amount_found"

    # Filter to only explicit-total candidates if they exist
    explicit = [
        c for c in candidates
        if ("|due_kw" in c["reason"] or "|total_kw" in c["reason"] or
            "|amount_kw" in c["reason"] or "next_line_kw" in c["reason"])
        and "|tier3_penalty" not in c["reason"]
        and "|change_reject" not in c["reason"]
        and "|cash_with_change_reject" not in c["reason"]
        and "|ref_id_penalty" not in c["reason"]
        and "|receipt_noise_reject" not in c["reason"]
        and "|url_reject" not in c["reason"]
        and "|prev_url_reject" not in c["reason"]
        and "|long_noise_reject" not in c["reason"]
        and "|very_long_digit_penalty" not in c["reason"]
    ]
    pool = explicit if explicit else candidates
    best = max(pool, key=lambda c: (c["score"], c["line"], c["value"]))

    score = best["score"]
    conf = 0.92 if score >= 200 else 0.82 if score >= 100 else 0.65 if score >= 50 else 0.45

    trace = f"best_score={score} value={best['value']} reason={best['reason']}\n"
    trace += f"total_kw_exists={total_kw_exists} due_kw_exists={due_kw_exists}\n"
    top5 = sorted(candidates, key=lambda c: c["score"], reverse=True)[:5]
    trace += "\n".join(
        f"  [{c['line']}] {c['text']!r} → {c['value']} score={c['score']} {c['reason']}"
        for c in top5
    )
    trace += "\n[TOKEN NORMALIZATION]\n" + "\n".join(
        f"  [{c['line']}] raw={c.get('raw_token')!r} normalized={c.get('normalized_token')!r} "
        f"value={c['value']} score={c['score']} reason={c['reason']}"
        for c in top5
    )
    rejected = [
        c for c in sorted(candidates, key=lambda c: c["score"])[:5]
        if c["score"] < 40 or "reject" in c["reason"] or "penalty" in c["reason"]
    ]
    if rejected:
        trace += "\n[REJECTED/LOW SCORE]\n" + "\n".join(
            f"  [{c['line']}] raw={c.get('raw_token')!r} normalized={c.get('normalized_token')!r} "
            f"text={c['text']!r} -> {c['value']} score={c['score']} {c['reason']}"
            for c in rejected
        )
    return float(best["value"]), conf, trace


def extract_amount_mbanking(
    ocr_lines: list[str],
) -> tuple[float, float, str]:
    """
    Extract main transaction amount from m-banking screenshot OCR lines.

    Prioritises large amounts near top/card area and near total/nominal keywords.
    Rejects ID/reference/PAN/token/date-like numbers.

    Returns:
        (amount, confidence, debug_trace)
    """
    doc_type = classify_document_type(ocr_lines)
    context_text = "\n".join(ocr_lines)
    # Sanitize OCR artifacts before normalising amounts
    norm = [_fix_ocr_digits(_sanitize_ocr_amount(ln)) for ln in ocr_lines]
    n = max(len(norm), 1)
    candidates: list[dict[str, Any]] = []

    _AMT_RE = re.compile(
        r"[\-–—]?\s*(?:Rp\.?|IDR)\s*([0-9OoIlSBb.,\s]+)",
        re.IGNORECASE,
    )

    def parse_idr(text: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for raw in _AMT_RE.findall(_fix_ocr_digits(text)):
            v = _parse_money_token(raw, idr_context=True, context=context_text, doc_type=doc_type)
            if v is not None and 1_000 <= v <= 1_000_000_000:
                out.append({
                    "value": v,
                    "raw_token": raw,
                    "normalized_token": _money_debug_normalized(
                        raw, idr_context=True, context=context_text, doc_type=doc_type
                    ),
                })
        return out

    def parse_loose(text: str) -> list[dict[str, Any]]:
        """Standalone thousand-separator numbers without Rp prefix."""
        out: list[dict[str, Any]] = []
        for m in re.finditer(r"\b(\d{1,3}(?:[.,]\d{3})+)\b", _fix_ocr_digits(text)):
            v = _parse_money_token(m.group(1), idr_context=True, context=context_text, doc_type=doc_type)
            if v is not None and 1_000 <= v <= 1_000_000_000:
                out.append({
                    "value": v,
                    "raw_token": m.group(1),
                    "normalized_token": _money_debug_normalized(
                        m.group(1), idr_context=True, context=context_text, doc_type=doc_type
                    ),
                })
        return out

    def add(idx: int, value: float, base: int, reason: str, raw_token: str = "", normalized_token: str = "") -> None:
        lower = norm[idx].lower()
        prev_context = " ".join(norm[max(0, idx - 2):idx]).lower()
        next_context = " ".join(norm[idx + 1:min(n, idx + 3)]).lower()
        window_context = f"{prev_context} {lower} {next_context}"
        score = base
        # Reject ID / reference / token lines
        if any(kw in lower for kw in _MBANKING_AMT_REJECT_KW):
            score -= 120
            reason += "|id_reject"
        if any(kw in prev_context for kw in _MBANKING_AMT_REJECT_KW) and not re.search(
            r"\b(total|nominal|jumlah|amount|tagihan|bayar|harga|charge|transfer)\b",
            window_context,
        ):
            score -= 140
            reason += "|prev_id_label_reject"
        if re.search(r"\b(id\s+transaksi|transaction\s+id|reference|no\.?\s*ref|"
                     r"merchant\s+pan|customer\s+pan|terminal\s+id|token|rrn|stan|"
                     r"trace|auth|approval|invoice|order\s*sn)\b", lower):
            score -= 120
            reason += "|ref_reject"
        if re.search(r"\b(customer\s+pan|merchant\s+pan|terminal\s+id|reference|"
                     r"transaction\s+id|source\s+of\s+fund|rekening|account)\b", window_context):
            if not re.search(r"\b(rp|idr|total|nominal|jumlah|amount|bayar)\b", window_context):
                score -= 100
                reason += "|technical_window_penalty"
        # Date lines
        if re.search(r"\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}\b|\b\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2}\b", norm[idx]):
            score -= 100
            reason += "|date_penalty"
        # Phone/long digit string
        if re.search(r"\b\d{10,}\b", re.sub(r"[.,]", "", norm[idx])):
            score -= 80
            reason += "|long_digit_penalty"
        if len(re.sub(r"\D", "", norm[idx])) >= 10 and not re.search(r"\b(rp|idr)\b", lower):
            score -= 220
            reason += "|pan_account_reject"
        # Main amount keywords boost
        if re.search(r"\b(total|nominal|jumlah|amount|tagihan|bayar|harga|charge|transfer)\b", lower):
            score += 70
            reason += "|main_kw"
        elif re.search(r"\b(total|nominal|jumlah|amount|tagihan|bayar|harga|charge|transfer)\b", prev_context):
            score += 55
            reason += "|prev_main_kw"
        # Rp/IDR prefix boost
        if re.search(r"(?:^|[\s(])\s*[-–—]?\s*(rp|idr)\b", lower):
            score += 50
            reason += "|rp_prefix"
        # Outgoing transfer (negative/debit)
        if re.search(r"[-–—]\s*(rp|idr)\b", lower):
            score += 30
            reason += "|outgoing"
        # Admin fee / balance → penalise
        if re.search(r"\b(biaya\s+admin|admin\s+fee|saldo|balance)\b", lower):
            score -= 60
            reason += "|fee_balance_penalty"
        # Prominence: upper area of screen
        if idx <= max(3, n // 3):
            score += 40
            reason += "|top_prominent"
        elif idx <= max(5, n // 2):
            score += 20
            reason += "|upper_half"
        if doc_type == "compact_qr_card":
            if idx <= max(4, n // 2):
                score += 30
                reason += "|compact_qr_top"
            if re.search(r"\b(qr\s+bayar|pembayaran\s+qr|pembayaran\s+qris|total|jumlah|nominal)\b", lower):
                score += 35
                reason += "|compact_qr_context"
        if _amount_doc_context(norm[idx], doc_type) and re.search(r"(?:rp|idr)", lower):
            score += 20
            reason += "|qr_currency_context"
        if re.search(r"\d+[.,]\d{4,}", ocr_lines[idx]) or (
            doc_type == "compact_qr_card" and re.search(r"\b\d{3}[.,]00[1-9]\b", _fix_ocr_digits(ocr_lines[idx]))
        ):
            score -= 10
            reason += "|artifact_corrected"
        candidates.append({
            "line": idx, "text": ocr_lines[idx],
            "raw_token": raw_token or ocr_lines[idx],
            "normalized_token": normalized_token or str(int(value) if float(value).is_integer() else value),
            "value": value, "score": score, "reason": reason,
        })

    for i, line in enumerate(norm):
        for token in parse_idr(line):
            add(i, token["value"], 80, "rp_same_line", token["raw_token"], token["normalized_token"])
        # Context keyword → lookahead
        lower = line.lower()
        if re.search(r"\b(total|nominal|jumlah|amount|tagihan|bayar|harga|charge|transfer)\b", lower):
            for j in range(i + 1, min(i + 4, n)):
                vals = parse_idr(norm[j]) or parse_loose(norm[j])
                for token in vals:
                    add(j, token["value"], 65, f"kw_lookahead[{i}]", token["raw_token"], token["normalized_token"])
                if vals:
                    break

    # Fallback
    if not candidates:
        full = "\n".join(norm)
        for token in parse_idr(full):
            add(0, token["value"], 40, "global_fallback", token["raw_token"], token["normalized_token"])
        if not candidates:
            for token in parse_loose(full):
                add(0, token["value"], 25, "global_loose_fallback", token["raw_token"], token["normalized_token"])

    if not candidates:
        return 0.0, 0.0, "no_amount_found"

    best = max(candidates, key=lambda c: (c["score"], c["value"]))
    score = best["score"]
    conf = 0.90 if score >= 150 else 0.78 if score >= 90 else 0.55 if score >= 40 else 0.30
    if "artifact_corrected" in best["reason"]:
        conf = min(conf, 0.72)
    elif "rp_prefix" not in best["reason"] and "global" in best["reason"]:
        conf = min(conf, 0.45)

    trace = f"best_score={score} value={best['value']} reason={best['reason']}\n"
    top5 = sorted(candidates, key=lambda c: c["score"], reverse=True)[:5]
    trace += "\n".join(
        f"  [{c['line']}] {c['text']!r} → {c['value']} score={c['score']} {c['reason']}"
        for c in top5
    )
    return float(best["value"]), conf, trace


# ============================================================
# MERCHANT PARSER
# ============================================================

# OCR typo → canonical name table
_MERCHANT_CANONICAL: dict[str, str] = {
    "alfamart": "Alfamart", "indomaret": "Indomaret", "alfamidi": "Alfamidi",
    "shopee": "Shopee", "tokopedia": "Tokopedia", "lazada": "Lazada",
    "mixue": "Mixue", "miegacoan": "Mie Gacoan",
}


def _clean_merchant_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _merchant_key_variants(key: str) -> set[str]:
    """Generate generic OCR-confusion variants for conservative fuzzy matching."""
    variants = {key}
    if not key:
        return variants
    variants.add(key.replace("0", "o").replace("1", "l"))
    variants.add(key.replace("rn", "m"))
    variants.add(re.sub(r"([a-z])\1{2,}", r"\1\1", key))
    if "t" in key:
        for idx, ch in enumerate(key):
            if ch == "t":
                variants.add(f"{key[:idx]}l{key[idx + 1:]}")
    return {variant for variant in variants if variant}


def _normalize_merchant_name(merchant: str) -> str:
    """Apply OCR correction table and return cleaned merchant name."""
    if not merchant or not merchant.strip():
        return merchant
    merchant = _trim_merchant_location(merchant)
    key = _clean_merchant_key(merchant)
    if key in _MERCHANT_CANONICAL:
        return _MERCHANT_CANONICAL[key]
    words = merchant.strip().split()
    if words:
        first_key = _clean_merchant_key(words[0])
        if first_key in _MERCHANT_CANONICAL:
            return _MERCHANT_CANONICAL[first_key]
    # Substring match for complete canonical tokens only.
    for ck, cn in _MERCHANT_CANONICAL.items():
        if ck in key and len(ck) >= 5:
            return cn
    best_score = 0.0
    best_name = ""
    if len(key) >= 6:
        for candidate_key in _merchant_key_variants(key):
            for canonical_key, canonical_name in _MERCHANT_CANONICAL.items():
                score = SequenceMatcher(None, candidate_key, canonical_key).ratio()
                if score > best_score:
                    best_score = score
                    best_name = canonical_name
    if best_score >= 0.88:
        return best_name
    return merchant.strip()


def _is_bad_merchant(text: str) -> bool:
    """True if text should never be used as a merchant name."""
    stripped = (text or "").strip()
    lower = stripped.lower()
    if not stripped or len(stripped) < 2:
        return True
    if lower in _MERCHANT_REJECT or lower in _CITY_REJECT:
        return True
    if any(lower.startswith(prefix) for prefix in _ADDRESS_PREFIXES):
        return True
    # Purely numeric / mostly-digit string
    digits = re.sub(r"\D", "", stripped)
    if len(digits) >= 10 and len(digits) / max(len(stripped), 1) > 0.55:
        return True
    # Reference/transaction ID variants such as "260303-DHM8-X6ACYO".
    if re.match(r"^[A-Z0-9]{3,12}(?:-[A-Z0-9]{2,12}){1,4}$", stripped, re.IGNORECASE):
        if any(c.isalpha() for c in stripped) and any(c.isdigit() for c in stripped):
            return True
    if _REF_LINE_RE.search(lower):
        return True
    if _RECEIPT_FOOTER_RE.search(lower):
        return True
    # Date pattern
    if re.search(r"\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}\b", stripped):
        return True
    if _try_parse_date(stripped):
        return True
    if re.fullmatch(
        r"\d{1,2}\s+(?:jan|januari|feb|februari|mar|maret|apr|april|may|mei|jun|juni|"
        r"jul|juli|aug|agu|ags|agustus|sep|sept|okt|oct|nov|des|dec)[a-z]*\.?\s+\d{2,4}",
        stripped,
        re.IGNORECASE,
    ):
        return True
    # Time pattern
    if re.search(r"\b\d{1,2}[:.]\d{2}(:\d{2})?\b", stripped):
        return True
    # Currency / amount
    if re.search(r"\b(rp|idr)\s*[\d.,Oo]+\b", lower):
        return True
    if re.match(r"^[\-–—]?\s*(rp|idr)\b", lower):
        return True
    # Location string: "CITY, POSTAL, COUNTRY"
    if re.match(r"^[A-Z][A-Z\s]+,\s*\d{4,6}\s*,?\s*(?:[A-Z]{2})?\s*$", stripped, re.I):
        return True
    if re.match(
        r"^(?:bandung|jakarta|bekasi|tangerang|surabaya|depok|bogor|semarang|"
        r"yogyakarta|jogja|medan|malang|denpasar|bali)\s*,?\s*id$",
        lower,
        re.I,
    ):
        return True
    # Alphanumeric-only code (transaction ID style)
    if re.match(r"^[A-Z0-9]{6,}$", stripped) and any(c.isdigit() for c in stripped):
        return True
    # 3-letter country code style
    if re.match(r"^[A-Z]{1,3}\d{1,4}$", stripped):
        return True
    # Phone status-bar / network-speed indicators: "0.2KBIs%", "33.3KBIsli", "89%"
    if re.search(
        r"\b(kb/?s|kbis|mb/?s|mbps)\b"       # network speed units
        r"|\d+\s*%\s*$"                         # percentage at end
        r"|^\d[\d.,]+\s*[kK][bB]",             # leading digit + KB prefix
        stripped,
        re.IGNORECASE,
    ):
        return True
    # Numeric prefix + QR/payment label: "89 QR Bayar", "12 Pembayaran"
    if re.match(r"^\d+\s+", stripped) and re.search(
        r"\b(qr\s+bayar|pembayaran|berhasil)\b", lower
    ):
        return True
    # Very few vowels for length ≥ 5 → probably OCR noise / code
    letters = [c for c in stripped if c.isalpha()]
    if len(letters) >= 6:
        vowels = sum(c.lower() in "aeiou" for c in letters)
        if vowels <= 1:
            return True
    if len(stripped) > 65 and re.search(
        r"\b(jl\.?|jalan|rt\.?|rw\.?|kec\.?|kel\.?|kota|kab\.?|kode\s+pos|no\.?)\b",
        lower,
    ):
        return True
    return False


def _is_item_table_line(line: str) -> bool:
    """True if this line looks like an item row (not a merchant header)."""
    lower = line.lower()
    # Quantity markers: "2 x", "2x", "x1", "x2", "qty", "pcs", "@"
    if re.search(r"\b\d+\s*x\b|\b\d+x\b|\bx\d+\b|\bqty\b|\bpcs\b|@", lower):
        return True
    # Leading-digit pattern: "1 ITEM NAME price"
    if re.match(r"^\s*\d+\s+[A-Za-z]", line.strip()):
        return True
    # Text + thousand-separated price at end: "KOPI SUSU  15,000"
    if re.search(r"[A-Za-z]{3,}.*\b\d{1,3}[.,]\d{3}\s*$", lower):
        return True
    if re.search(r"\b\d{1,3}[.,]\d{3}\b", lower) and not re.search(
        r"\b(grand\s+total|total|amount\s+due|total\s+due|balance\s+due|due|bayar|tunai)\b",
        lower,
    ):
        return True
    # Text + zero / free-item price at end: "SENDOK BEBEK  0", "PLASTIK 25  0"
    # Matches "ITEM NAME  0" but NOT "STORE NO.  1000" (0 must be standalone)
    if re.search(r"[A-Za-z]{3,}.*\s+0(?:\.0+)?\s*$", lower):
        return True
    if _RECEIPT_ITEM_WORD_RE.search(lower) and not _RECEIPT_MERCHANT_CUE_RE.search(lower):
        words = re.findall(r"[a-z]+", lower)
        if len(words) <= 5:
            return True
    return False


def _is_receipt_merchant_noise(line: str, *, normalized: str = "") -> bool:
    """Reject receipt merchant candidates that look like menu/body/footer noise."""
    clean = re.sub(r"\s+", " ", line or "").strip(" ,.;:-")
    lower = clean.lower()
    norm_lower = (normalized or clean).lower()
    if not clean:
        return True
    if _RECEIPT_FOOTER_RE.search(lower) or _RECEIPT_AMOUNT_REJECT_RE.search(lower):
        return True
    if re.search(r"(?:https?://|www\.|\.com|\.co\.id|link|url|/f/)", lower):
        return True
    if _is_item_table_line(clean):
        return True
    if re.search(r"\b\d+\s*x\b|\b\d+x\b|\bx\d+\b|^\s*\d+\s+[A-Za-z]", lower):
        return True
    if re.match(r"^\s*\d+\s*(?:x|X)?\s+[A-Za-z]", clean):
        return True
    if re.search(r"\b(rp|idr)?\s*\d{1,3}[.,]\d{3}\b", lower):
        return True
    words = re.findall(r"[a-z]+", lower)
    if len(clean) <= 4:
        return True
    if _GENERIC_SINGLE_WORD_ITEM_RE.fullmatch(lower) and not _RECEIPT_MERCHANT_CUE_RE.search(norm_lower):
        return True
    letters = re.sub(r"[^A-Za-z]", "", clean)
    non_letters = re.sub(r"[A-Za-z\s]", "", clean)
    if len(non_letters) / max(len(clean), 1) > 0.35 and not _RECEIPT_MERCHANT_CUE_RE.search(norm_lower):
        return True
    if len(letters) <= 4 and not _RECEIPT_MERCHANT_CUE_RE.search(norm_lower):
        return True
    if len(letters) >= 5:
        vowels = sum(ch.lower() in "aeiou" for ch in letters)
        if vowels <= 1 or vowels / max(len(letters), 1) < 0.18:
            return True
    if len(letters) >= 3:
        unique_letters = len(set(letters.lower()))
        if unique_letters <= 2 and not _RECEIPT_MERCHANT_CUE_RE.search(norm_lower):
            return True
    if _RECEIPT_ITEM_WORD_RE.search(lower) and not _RECEIPT_MERCHANT_CUE_RE.search(norm_lower):
        canonical_key = _clean_merchant_key(normalized or clean)
        if canonical_key not in _MERCHANT_CANONICAL:
            return True
    if len(words) == 1:
        key = _clean_merchant_key(normalized or clean)
        if key not in _MERCHANT_CANONICAL and not _RECEIPT_MERCHANT_CUE_RE.search(norm_lower):
            return True
    return False


def _find_item_table_start(lines: list[str]) -> int:
    """Index of first line that looks like item/total section."""
    kw = ("subtotal", "sub total", "total", "tax", "ppn", "pb1", "service", "tunai", "qty")
    for i, line in enumerate(lines):
        lower = _fix_ocr_digits(line).lower()
        if any(k in lower for k in kw):
            return max(0, i - 2)
        if re.search(r"\b(qty|item|price|amount)\b", lower):
            return i
        if _is_item_table_line(line):
            return i
    return len(lines)


def extract_merchant_receipt(
    ocr_lines: list[str],
) -> tuple[str, float, str]:
    """
    Extract merchant from receipt OCR lines.
    Only looks in header area (above item table).

    Returns (merchant_name, confidence, debug_trace)
    """
    table_start = _find_item_table_start(ocr_lines)
    # Header = top 20-30% only, and never past the item/total section.
    top_limit = max(3, min(10, max(1, (len(ocr_lines) + 3) // 4)))
    header_limit = min(len(ocr_lines), top_limit)
    if table_start < len(ocr_lines):
        header_limit = min(header_limit, max(0, table_start))
    header_lines = ocr_lines[:header_limit]

    trace = f"table_start={table_start} header_limit={header_limit}\n"
    candidates: list[dict[str, Any]] = []

    skip_kw = (
        "subtotal", "total", "tax", "ppn", "pb1", "service",
        "tunai", "payment", "debit", "qris", "qty", "cashier",
        "kasir", "operator", "receipt", "rcpt", "table", "guest",
        "shopping", "groceries", "category", "belanja",
        # Hotel / document headers
        "folio", "guest folio", "account statement", "your account",
        # Sentence-like footer markers
        "satisfaction", "reservations", "return policy",
        "our pleasure", "hope you", "unhappy",
        "important to us", "customer service",
    )
    # Regex for OCR-noise "RECEIPT" variants: "RECEIPI", "RECE IPT", "RECE", "RCPT"
    _receipt_noise_re = re.compile(
        r"\brece\b"              # split OCR "RECE IPT"
        r"|\breceip[a-z]{0,3}\b"  # "receipi", "receipt", "receipts"
        r"|\brcpt\b",
        re.IGNORECASE,
    )
    # Sentence detector: 5+ function words → this is running text, not a store name
    _sentence_words = frozenset({
        "your", "our", "the", "and", "but", "are", "were", "will",
        "have", "has", "with", "for", "that", "this", "from", "they",
        "you", "we", "it", "not", "any", "all", "been", "may",
        "if", "of", "to", "in", "is", "or",
    })

    for i, line in enumerate(header_lines):
        stripped = line.strip()
        if len(stripped) < 3:
            continue
        lower = stripped.lower()
        if any(lower.startswith(p) for p in _ADDRESS_PREFIXES):
            trace += f"  skip[{i}] address prefix: {stripped!r}\n"
            continue
        if any(kw in lower for kw in skip_kw):
            trace += f"  skip[{i}] receipt keyword: {stripped!r}\n"
            continue
        if any(s in lower for s in ("http", "www.", "@", ".com")):
            trace += f"  skip[{i}] url: {stripped!r}\n"
            continue
        if not re.search(r"[A-Za-z]{3,}", stripped):
            trace += f"  skip[{i}] no letters: {stripped!r}\n"
            continue
        # Lines too long to be a store name (> 60 chars → likely body text)
        if len(stripped) > 60:
            trace += f"  skip[{i}] too_long ({len(stripped)}): {stripped[:40]!r}\n"
            continue
        # Line ends in ":" → label/header or OCR suffix, not a store name
        if _GENERIC_SINGLE_WORD_ITEM_RE.fullmatch(stripped.strip()) and not _RECEIPT_MERCHANT_CUE_RE.search(stripped):
            trace += f"  skip[{i}] generic_single_item: {stripped!r}\n"
            continue
        if stripped.rstrip().endswith(":"):
            trace += f"  skip[{i}] label_colon: {stripped!r}\n"
            continue
        # OCR-noise receipt variants: "RECEIPI", "RECE IPT", etc.
        if _receipt_noise_re.search(lower):
            trace += f"  skip[{i}] receipt_noise: {stripped!r}\n"
            continue
        # Sentence-like text: too many function words → body text, not store name
        # Also catches short disclaimers like "Not VaLTD FOR" (2 fn-words in ≤4 words)
        words_lower = re.findall(r"[a-z]+", lower)
        sent_word_count = sum(1 for w in words_lower if w in _sentence_words)
        _is_sentence = (
            (sent_word_count >= 3 and len(words_lower) > 3)
            or (sent_word_count >= 2 and len(words_lower) <= 4)
        )
        if _is_sentence:
            trace += f"  skip[{i}] sentence_text ({sent_word_count}/{len(words_lower)}): {stripped!r}\n"
            continue
        normalized = _normalize_merchant_name(stripped)
        normalized = _trim_merchant_location(normalized)
        if _is_receipt_merchant_noise(stripped, normalized=normalized):
            trace += f"  skip[{i}] receipt_item_or_noise: {stripped!r}\n"
            continue
        if _is_bad_merchant(normalized):
            trace += f"  skip[{i}] bad_merchant: {stripped!r}\n"
            continue

        # Skip lines that look like item rows (text + inline price)
        if _is_item_table_line(stripped):
            trace += f"  skip[{i}] item_table_line: {stripped!r}\n"
            continue

        # Skip if the first non-empty following line is a standalone price or
        # an item-table line (qty/pcs pattern), which indicates THIS line is
        # an item name rather than a merchant header.
        _next_is_price = False
        for _nxt in range(i + 1, min(i + 3, len(header_lines))):
            _nl = header_lines[_nxt].strip()
            if not _nl:
                continue
            # Pure-numeric line (no letters) that parses as a monetary amount
            if (not re.search(r"[A-Za-z]", _nl)
                    and not _try_parse_date(_nl)
                    and bool(_extract_amounts_from_line(_nl))):
                _next_is_price = True
            # Next line is an item row with qty/pcs/x indicator.
            # Only reject when current candidate is a single unknown word
            # (not a known store like Alfamart whose normalization changes it).
            elif (_is_item_table_line(_nl)
                  and not re.search(r"\b(subtotal|total|tax|service)\b", _nl.lower())
                  and len(stripped.split()) == 1       # single-word candidate
                  and normalized == stripped.strip()):  # not a known canonical store
                _next_is_price = True
            break  # only check first non-empty following line
        if _next_is_price:
            trace += f"  skip[{i}] item_row (price_or_qty_below): {stripped!r}\n"
            continue

        score = 100 - i * 8
        if i <= 2:
            score += 30
        letters = sum(c.isalpha() for c in stripped)
        digits = sum(c.isdigit() for c in stripped)
        if digits > letters:
            score -= 40
        vowels = sum(c.lower() in "aeiou" for c in stripped if c.isalpha())
        if letters >= 5 and vowels >= 2:
            score += 10
        candidates.append({"text": normalized, "score": score, "line": i})
        trace += f"  cand[{i}] {stripped!r} → {normalized!r} score={score}\n"

    if not candidates:
        return "Merchant tidak terdeteksi", 0.20, trace + "result: no_candidate"

    best = max(candidates, key=lambda c: c["score"])
    if best["score"] < 30:
        return "Merchant tidak terdeteksi", 0.20, trace + f"result: low_score={best['score']}"
    conf = 0.88 if best["score"] >= 80 else 0.70 if best["score"] >= 50 else 0.45
    return best["text"], conf, trace + f"result={best['text']!r} score={best['score']}"


def extract_merchant_mbanking(
    ocr_lines: list[str],
) -> tuple[str, float, str]:
    """
    Extract recipient/merchant from m-banking screenshot OCR lines.

    Strategy:
    1. Inline patterns: "Pembayaran QR ke X", "Payment to X", "Bayar ke X"
    2. Keyword-anchored: look for label lines, take value after
    3. ALL-CAPS candidate (QRIS merchant names are usually ALL-CAPS)
    4. First text-rich non-noise line

    Returns (merchant_name, confidence, debug_trace)
    """
    trace: list[str] = []

    _LABEL_KW = (
        "bayar ke", "bayar kepada", "payment to", "pembayaran ke",
        "merchant", "nama merchant", "nama toko", "nama usaha",
        "penerima", "kepada", "tujuan", "recipient",
        "paid to", "store", "diterima oleh",
    )
    doc_type = classify_document_type(ocr_lines)

    def good(raw: str) -> str | None:
        clean = re.sub(r"\s+", " ", raw).strip(" :;-")
        # "ke X" / "to X" / "bayar ke X" — strip even without separator
        # Strip leading navigation particles — require whitespace/separator after
        # "ke" and "to" to avoid stripping the prefix of merchant names like "Keikpop"
        clean = re.sub(r"^(?:ke|to)\s+", "", clean, flags=re.IGNORECASE)
        clean = re.sub(
            r"^(?:bayar\s+ke|pembayaran\s+ke|payment\s+to|paid\s+to|"
            r"pembayaran\s+qr\s+ke|qr\s+bayar\s+ke|penerima)\s*[:\-]?\s*",
            "",
            clean,
            flags=re.IGNORECASE,
        )
        # "Merchant: X" / "Acquirer: X" — strip label prefix when separator present
        # (separator required so "MERCHANT ABC" stays intact)
        clean = re.sub(
            r"^(?:merchant|nama\s+merchant|nama\s+toko|nama\s+usaha|"
            r"acquirer(?:\s+name)?|terminal\s+id|customer\s+pan|merchant\s+pan)\s*[:\-]\s*",
            "", clean, flags=re.IGNORECASE,
        )
        # Strip trailing location suffix " - CITY" — keep merchant base name
        # But preserve if the whole name IS the location (i.e. it's the actual store name)
        # Only strip if city name alone is in _CITY_REJECT
        clean = re.sub(r",\s*\d{6,}.*$", "", clean)
        clean = re.sub(r"\s+\d{8,}.*$", "", clean)
        clean = _trim_merchant_location(clean)
        clean = clean.strip(" ,.;:-")
        if len(clean) < 2 or _is_bad_merchant(clean):
            return None
        return clean

    def amount_line(text: str) -> bool:
        return bool(re.search(r"[-â€“â€”]?\s*(?:rp|idr)\.?\s*[\d.,OoIlSBb]+", _fix_ocr_digits(text), re.I))

    def reference_or_label(text: str) -> bool:
        lower = text.strip().lower()
        return (
            not lower
            or lower in _MERCHANT_REJECT
            or bool(_REF_LINE_RE.search(lower))
            or bool(re.search(
                r"\b(transaction|transaksi|details?|rincian|date|tanggal|time|waktu|"
                r"payment method|metode pembayaran|source of fund|sumber dana|"
                r"customer pan|merchant pan|terminal id|acquirer|status|share|bagikan)\b",
                lower,
            ))
        )

    def readable_merchant_line(text: str) -> str | None:
        s = text.strip()
        if amount_line(s) or reference_or_label(s):
            return None
        if len(s) > 70 or not re.search(r"[A-Za-z]{3,}", s):
            return None
        return good(s)

    if doc_type == "mbanking_transaction_detail":
        for i, line in enumerate(ocr_lines[: max(8, len(ocr_lines) // 2)]):
            if not amount_line(line):
                continue
            title_parts: list[str] = []
            for j in range(i + 1, min(i + 5, len(ocr_lines))):
                if reference_or_label(ocr_lines[j]):
                    if title_parts:
                        break
                    continue
                cand = readable_merchant_line(ocr_lines[j])
                if cand:
                    title_parts.append(cand)
                    if len(title_parts) >= 2:
                        break
            if title_parts:
                merged = re.sub(r"\s+", " ", " ".join(title_parts)).strip()
                cand = good(merged) or title_parts[0]
                conf = 0.86 if len(cand.split()) >= 2 else 0.62
                trace.append(f"strat_detail_title_after_amount[{i}]: {cand!r}")
                return cand, conf, "\n".join(trace)

        top_boundary = next(
            (i for i, line in enumerate(ocr_lines[: min(len(ocr_lines), 18)])
             if _REF_LINE_RE.search(line.lower())),
            min(len(ocr_lines), 10),
        )
        top_cands = [
            c for c in (readable_merchant_line(line) for line in ocr_lines[:top_boundary])
            if c
        ]
        if top_cands:
            best = max(top_cands, key=lambda c: (len(c.split()), len(c)))
            trace.append(f"strat_detail_top_card: {best!r}")
            return best, 0.74, "\n".join(trace)

    # ── Strategy 0: inline "QR Bayar ke X" / "Payment to X" on same line ──────
    for line in ocr_lines:
        m = re.search(r"^\s*ke\s+(.+)", line, re.IGNORECASE)
        if m:
            c = good(m.group(1))
            if c:
                trace.append(f"strat0_ke_same_line: {c!r}")
                return c, 0.84 if doc_type == "compact_qr_card" else 0.76, "\n".join(trace)
        m = re.search(r"(?:pembayaran\s+qr|qr\s+bayar)\s+ke\s+(.+)", line, re.IGNORECASE)
        if m:
            c = good(m.group(1))
            if c:
                trace.append(f"strat0_qr_ke: {c!r}")
                return c, 0.88, "\n".join(trace)
        m = re.search(r"payment\s+to\s+(.+)", line, re.IGNORECASE)
        if m:
            c = good(m.group(1))
            if c:
                trace.append(f"strat0_payment_to: {c!r}")
                return c, 0.88, "\n".join(trace)
        # "Bayar ke" on same line as merchant
        m = re.search(r"(?:bayar\s+ke|kepada|penerima)\s*[:\-]?\s*(.+)", line, re.IGNORECASE)
        if m:
            c = good(m.group(1))
            if c:
                trace.append(f"strat0_inline_label: {c!r}")
                return c, 0.85, "\n".join(trace)

    # ── Strategy 0b: QR Bayar split across lines ─────────────────────────────
    # Cropped QR screenshots often have:
    #   line[i]   = "QR Bayar"  or  "Pembayaran QR"
    #   line[i+1] = "-IDR 49.999"   (amount)
    #   line[i+2] = "ke KIMUKATSU CENTRAL PARK MA"
    for i, line in enumerate(ocr_lines):
        lower_i = line.strip().lower()
        if re.search(r"\b(qr\s+bayar|pembayaran\s+qr)\b", lower_i):
            for j in range(i + 1, min(i + 6, len(ocr_lines))):
                cand_line = ocr_lines[j].strip()
                # Skip amount lines (–IDR / Rp prefix)
                if re.search(r"(?:rp|idr)\.?\s*[\d.,]+", cand_line, re.IGNORECASE):
                    continue
                # "ke MERCHANT" line — strip the "ke " prefix
                ke_m = re.match(r"^ke\s+(.+)", cand_line, re.IGNORECASE)
                if ke_m:
                    c = good(ke_m.group(1))
                    if c:
                        trace.append(f"strat0b_qr_split_ke[{i}→{j}]: {c!r}")
                        return c, 0.85, "\n".join(trace)
                if re.fullmatch(r"ke|to", cand_line, re.IGNORECASE):
                    for k in range(j + 1, min(j + 4, len(ocr_lines))):
                        c = readable_merchant_line(ocr_lines[k])
                        if c:
                            trace.append(f"strat0b_qr_separate_ke[{j}->{k}]: {c!r}")
                            return c, 0.82, "\n".join(trace)
                # Non-label, non-amount text line after QR marker
                if (not re.search(r"(?:rp|idr|qr|bayar|pembayaran|berhasil|"
                                   r"transaksi|rincian|detail)", cand_line, re.IGNORECASE)
                        and re.search(r"[A-Za-z]{4,}", cand_line)):
                    c = good(cand_line)
                    if c:
                        trace.append(f"strat0b_qr_split_text[{i}→{j}]: {c!r}")
                        return c, 0.80, "\n".join(trace)

    # ── Strategy 1: keyword-anchored (label on one line, value on next) ──────
    for i, line in enumerate(ocr_lines):
        lower = line.lower()
        if any(kw in lower for kw in _LABEL_KW):
            for j in range(i + 1, min(i + 8, len(ocr_lines))):
                cand = ocr_lines[j].strip()
                cand_lower = cand.lower()
                if not cand or cand_lower in _MERCHANT_REJECT:
                    continue
                if _is_bad_merchant(cand):
                    trace.append(f"strat1_skip[{j}] bad: {cand!r}")
                    continue
                c = good(cand)
                if c:
                    trace.append(f"strat1_kw_anchor[{i}→{j}]: {c!r}")
                    return c, 0.82, "\n".join(trace)

    # ── Strategy 2: ALL-CAPS candidate (typical QRIS merchant name) ──────────
    _skip_re = re.compile(
        r"^\s*("
        # Time / numeric-only lines
        r"[\d:.,\s]+(wib|wita|wit|am|pm)?"
        # Amount lines (Rp/IDR prefix)
        r"|[-–—]?\s*(?:rp|idr)\.?\s*[\d.,Oo]+"
        # Network speed / phone status bar: "0.2KBIs%", "33.3 KB/s"
        r"|[\d.,]+\s*(?:kb/?s|kbis|mb/?s|mbps|%)"
        # Payment/QR UI labels
        r"|qris\s*$|qr\s+bayar|qr\s*code"
        r"|pembayaran\s*(?:berhasil|qr)?"
        r"|transaksi\s*berhasil|transaction\s*details?"
        r"|rincian|rincian\s+pembayaran|rincian\s+transaksi"
        r")\s*$",
        re.IGNORECASE,
    )
    upper_cands: list[str] = []
    all_cands: list[str] = []
    for line in ocr_lines:
        s = line.strip()
        if not s or s.lower() in _MERCHANT_REJECT or s.lower() in _CITY_REJECT:
            continue
        if _is_bad_merchant(s):
            continue
        if _skip_re.match(_fix_ocr_digits(s)):
            continue
        if not re.search(r"[A-Za-z]{3,}", s):
            continue
        c = good(s)
        if not c:
            continue
        if s == s.upper() and re.search(r"[A-Z]", s):
            upper_cands.append(c)
        else:
            all_cands.append(c)

    if upper_cands:
        best = upper_cands[0]
        trace.append(f"strat2_allcaps: {best!r}")
        return best, 0.72, "\n".join(trace)

    if all_cands:
        best = all_cands[0]
        trace.append(f"strat3_first_text: {best!r}")
        return best, 0.55, "\n".join(trace)

    trace.append("no_merchant_found")
    return "Merchant tidak terdeteksi", 0.20, "\n".join(trace)


# ============================================================
# MAIN POSTPROCESSOR ENTRY POINT
# ============================================================

def postprocess(
    ocr_lines: list[str],
    route_type: str,
    donut_result: dict[str, Any] | None = None,
    existing_merchant: str = "",
    existing_amount: float = 0.0,
    existing_date: str = "",
) -> dict[str, Any]:
    """
    Unified postprocessor. Accepts raw OCR lines and returns clean fields.

    Args:
        ocr_lines:        Raw OCR lines from EasyOCR (already merged/split-fixed).
        route_type:       "receipt" or "screenshot".
        donut_result:     Optional DONUT parse result dict.
        existing_*:       Fields already extracted by primary parser (used as hints).

    Returns dict with keys:
        merchant, amount, date, field_confidence, warnings, debug_trace
    """
    is_screenshot = route_type == "screenshot"
    trace_parts: list[str] = [f"[POSTPROCESSOR] route={route_type} lines={len(ocr_lines)}"]
    warnings: list[str] = []

    # ---- Document-type classification ----------------------------------------
    doc_type = classify_document_type(ocr_lines)
    trace_parts.append(f"  doc_type={doc_type}")
    is_compact_qr = (doc_type == "compact_qr_card")

    # ---- Currency detection (receipt only; screenshots are always IDR context) ----
    currency = "IDR"
    if not is_screenshot:
        currency = detect_receipt_currency(ocr_lines)
        if currency != "IDR":
            trace_parts.append(f"  currency: detected={currency} (non-IDR)")
            warnings.append(
                f"Mata uang non-IDR terdeteksi ({currency}). "
                "Nominal ditampilkan dalam mata uang asli."
            )
        else:
            trace_parts.append("  currency: IDR")

    # ---- Merchant ----
    if is_screenshot:
        merchant, merch_conf, merch_trace = extract_merchant_mbanking(ocr_lines)
    else:
        merchant, merch_conf, merch_trace = extract_merchant_receipt(ocr_lines)

    # Validate existing merchant from primary parser too. The raw parser can be
    # right while the postprocessor lands on a UI label or ID; keep the valid
    # raw value in that case, and prefer a fuller raw title over a weak short one.
    if existing_merchant and not _is_bad_merchant(existing_merchant):
        existing_norm = _trim_merchant_location(_normalize_merchant_name(existing_merchant))
        if existing_norm and not _is_bad_merchant(existing_norm):
            merchant_is_bad = _is_bad_merchant(merchant) or merchant == "Merchant tidak terdeteksi"
            existing_is_fuller = (
                merch_conf < 0.72
                and len(existing_norm.split()) > len((merchant or "").split())
                and len(existing_norm) >= len(merchant or "")
            )
            if merchant_is_bad or merch_conf < 0.55 or existing_is_fuller:
                merchant = existing_norm
                merch_conf = max(merch_conf, 0.62 if is_screenshot else 0.50)
                trace_parts.append(f"  merchant: kept existing={existing_norm!r} (postprocessor invalid/weak)")
            else:
                trace_parts.append(f"  merchant: postprocessor={merchant!r} conf={merch_conf}")
        else:
            trace_parts.append(f"  merchant: existing was bad, using postprocessor={merchant!r}")
    else:
        trace_parts.append(f"  merchant: postprocessor={merchant!r} conf={merch_conf}")

    # ---- Confidence calibration: lower for single-word / only-candidate merchant ----
    if merchant and merchant != "Merchant tidak terdeteksi":
        words = merchant.split()
        if len(words) == 1 and merch_conf > 0.55:
            # Single-word merchant is ambiguous — could be item name or store name
            merch_conf = min(merch_conf, 0.55)
            trace_parts.append(f"  merchant: conf capped to 0.55 (single word)")

    trace_parts.append("[MERCHANT TRACE]")
    trace_parts.append(merch_trace)

    # ---- Amount ----
    if is_screenshot:
        amount, amt_conf, amt_trace = extract_amount_mbanking(ocr_lines)
    else:
        amount, amt_conf, amt_trace = extract_amount_receipt(ocr_lines)

    # Try DONUT total as fallback for receipts
    donut_amount = 0.0
    if not is_screenshot and donut_result:
        total_data = donut_result.get("total", {})
        if isinstance(total_data, dict):
            raw_total = total_data.get("total_price", "")
        else:
            raw_total = ""
        if raw_total:
            idr_ctx = (currency == "IDR")
            parsed = _parse_money_token(str(raw_total), idr_context=idr_ctx)
            if parsed and parsed > 0:
                donut_amount = parsed

    if amount <= 0 and donut_amount > 0:
        amount = donut_amount
        amt_conf = 0.70
        trace_parts.append(f"  amount: using donut_fallback={donut_amount}")
    elif amount > 0 and donut_amount > 0 and amt_conf < 0.65:
        # DONUT total as cross-check — if they agree, boost confidence
        if abs(amount - donut_amount) / max(amount, 1) < 0.05:
            amt_conf = min(0.95, amt_conf + 0.15)
            trace_parts.append(f"  amount: donut_agrees amount={amount}")
        else:
            trace_parts.append(f"  amount: donut_differs amount={amount} donut={donut_amount}")

    # Use existing amount if postprocessor found nothing
    if amount <= 0 and existing_amount > 0:
        amount = existing_amount
        amt_conf = 0.40
        trace_parts.append(f"  amount: kept existing={existing_amount}")

    # ---- Confidence calibration: per evidence quality ---------------------
    # Non-IDR decimal handling may not be perfect
    if currency != "IDR" and amt_conf > 0.65:
        amt_conf = min(amt_conf, 0.65)
        trace_parts.append("  amount: conf capped 0.65 (non-IDR currency)")
    # Compact QR cards have clean amounts but weak context for other fields
    if is_compact_qr and amount > 0 and amt_conf > 0.75:
        # Amount is probably reliable but give moderate confidence since
        # OCR on compact cards is more error-prone
        amt_conf = min(amt_conf, 0.80)
        trace_parts.append("  amount: conf capped 0.80 (compact_qr_card)")

    trace_parts.append("[AMOUNT TRACE]")
    trace_parts.append(amt_trace)

    # ---- Date ----
    date, date_conf, date_trace = extract_date(ocr_lines, route_type)

    # Compact QR cards typically do NOT show a full date — do not carry over
    # the existing_date if this looks like a cropped QR card without date text,
    # because that risks surfacing a stale date from a previous extraction.
    if is_compact_qr and not date:
        # Only keep existing_date when there is explicit date text nearby
        has_date_label = any(
            kw in " ".join(ocr_lines).lower()
            for kw in ("tanggal", "date", "waktu", "time")
        )
        if existing_date and has_date_label:
            date = existing_date
            date_conf = 0.40
            trace_parts.append(f"  date: kept existing (compact_qr with label)={existing_date!r}")
        else:
            date_conf = 0.0  # no date visible, return empty
            trace_parts.append("  date: compact_qr_card — no date label found, skipping")
    elif not date and existing_date:
        date = existing_date
        date_conf = 0.50
        trace_parts.append(f"  date: kept existing={existing_date!r}")
    else:
        trace_parts.append(f"  date: postprocessor={date!r} conf={date_conf}")

    trace_parts.append("[DATE TRACE]")
    trace_parts.append(date_trace)

    # ---- Warnings ----
    if merch_conf < 0.45:
        warnings.append("Merchant belum terdeteksi. Periksa dan isi manual.")
    if amt_conf < 0.45:
        warnings.append("Nominal tidak yakin. Periksa kembali.")
    if date_conf < 0.45:
        warnings.append("Tanggal tidak terdeteksi. Isi manual.")
    if any(c < 0.55 for c in (merch_conf, amt_conf, date_conf)):
        warnings.append("Beberapa data perlu diperiksa kembali.")

    return {
        "merchant": merchant,
        "amount": amount,
        "date": date,
        "currency": currency,
        "field_confidence": {
            "merchant": merch_conf,
            "amount": amt_conf,
            "date": date_conf,
        },
        "warnings": warnings,
        "debug_trace": "\n".join(trace_parts),
    }
