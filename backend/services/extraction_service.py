"""Extraction service â€” wraps existing OCR/parser modules for FastAPI.

Intentionally minimal: reuses MBankingParser, ReceiptParser,
extraction_postprocessor.postprocess, and HybridCategoryClassifier without
rewriting OCR logic. All bug-fixes here are safe normalisations only.
"""
from __future__ import annotations

import logging
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Ensure project root is on sys.path so existing modules are importable
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR = PROJECT_ROOT / "models" / "indobert" / "run1" / "final"
ENABLE_DONUT_FALLBACK = False
MAX_EXTRACTION_SECONDS = 30.0

CATEGORY_DISPLAY: dict[str, str] = {
    "makanan_minuman": "Makanan & Minuman",
    "transportasi":    "Transportasi",
    "belanja":         "Belanja & Retail",
    "hiburan":         "Hiburan & Wisata",
    "kesehatan":       "Kesehatan",
    "pendidikan":      "Pendidikan",
    "tagihan":         "Tagihan & Utilitas",
    "lainnya":         "Lainnya",
}

SCREENSHOT_INTERNAL_TYPES = {
    "mbanking_transaction_detail",
    "compact_qr",
    "compact_qr_card",
    "qris",
    "ewallet",
    "ewallet_receipt_screen",
    "payment_screenshot",
    "screenshot",
}

RECEIPT_INTERNAL_TYPES = {"receipt", "photo_receipt", "shopping_receipt"}

DOC_TYPE_LABEL = {
    "receipt": "Struk Belanja",
    "screenshot": "Screenshot Pembayaran",
    "unknown": "Perlu dicek manual",
}

# Address-like separators that signal end of a merchant name
_MERCHANT_ADDR_SEPS = (
    r"\bjl\b", r"\bji\b", r"\bjalan\b", r"\bkec\b", r"\bkel\b", r"\bkab\b",
    r"\brt\b", r"\brw\b", r"\bno\b", r"\blt\b", r"\bblok\b",
    r"\bgedung\b", r"\bruko\b", r"\bkomplek\b", r"\bperumahan\b",
)
_MERCHANT_ADDR_RE = re.compile(
    r"[,\s]+(?:" + "|".join(_MERCHANT_ADDR_SEPS) + r")\b",
    re.IGNORECASE,
)
_MERCHANT_HARD_CUT_RE = re.compile(
    r"(\s+J[lI]\.?\s+|\s+Jl\.?\s+|\s+Ji\.?\s+|\s+Jalan\s+|,\s*J[lI]\b|,\s*Jl\b|\s+Kec\.\s+|\s+Kecamatan\s+|"
    r"\s+Kota\s+|\s+Kabupaten\s+|\s+RT\b|\s+RW\b|\s+No\.\s+|"
    r"\s+Blok\s+|\s+Ruko\s+)",
    re.IGNORECASE,
)

# Patterns that look like transaction IDs / status text, not merchant names
_MERCHANT_JUNK_RE = re.compile(
    r"^(pembayaran berhasil|transaksi berhasil|transaction details?|"
    r"rincian (transaksi|pembayaran)|payment success|berhasil|success|"
    r"selesai|completed|done|ok)$",
    re.IGNORECASE,
)
_MERCHANT_FORBIDDEN_RE = re.compile(
    r"\b(pembayaran berhasil|transaksi berhasil|berhasil|success|selesai|"
    r"transaction details?|from|to|source of fund|payment method|acquirer|"
    r"reference no|reference number|transaction id|id transaksi|customer pan|"
    r"merchant pan|terminal id|total transaksi|total|jumlah|nominal|kategori|"
    r"detail transaksi|rincian transaksi|rincian pembayaran)\b",
    re.IGNORECASE,
)

# Screenshot auto-detection keywords
_SC_STRONG = frozenset({
    "qr bayar", "qris", "transaksi berhasil", "pembayaran berhasil",
    "rincian pembayaran", "detail transaksi", "bayar ke",
    "transaction details", "payment details", "payment successful",
    "reference no", "reference number", "transaction id", "terminal id",
    "source of fund", "main pocket", "acquirer name", "bank",
    "customer pan", "merchant pan", "rrn", "stan",
    "nominal transaksi", "tanggal transaksi", "date and time",
    "transfer", "pembayaran qr", "pembayaran qris",
    "gopay", "ovo", "dana", "shopeepay", "linkaja", "blu", "seabank",
    "bank jago", "bank mandiri", "bank bca", "bank bni", "bank bri",
})
_RC_STRONG = frozenset({
    "kasir", "cashier", "subtotal", "kembalian", "tunai", "qty",
    "grand total", "total belanja", "total bayar", "receipt",
    "number of items", "tax", "ppn", "pb1", "service charge",
    "terima kasih", "struk", "nota", "nomor struk", "no struk",
    "guest folio", "folio", "balance due", "charges", "credits",
    "room", "receipt no",
    "alfamart", "indomaret", "alfamidi",
    "minimarket", "supermarket",
})

_CATEGORY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tagihan", (
        "pln", "listrik", "pdam", "internet", "wifi", "pulsa", "paket data",
        "telkom", "youtube music", "spotify", "netflix", "subscription",
    )),
    ("transportasi", ("spbu", "shell", "parkir", "parking", "tol", "toll", "grab", "gojek", "kereta", "travel")),
    ("belanja", (
        "alfamart", "indomaret", "alfamidi", "minimarket", "supermarket",
        "shopee", "tokopedia", "lazada", "store", "mart", "retail", "toko",
        "aksesoris", "accessories",
    )),
    ("makanan_minuman", (
        "coffee", "kopi", "cafe", "resto", "restaurant", "restoran", "makanan",
        "minuman", "mie", "ayam", "bakso", "mixue", "gacoan", "martabak",
        "pizza", "mozzarella", "dapur", "tahu", "kantin", "warung",
        "warmindo", "warteg", "kedai", "nasi", "roti", "bread", "bakery",
        "cake", "cakes",
    )),
    ("lainnya", ("laundry", "dry clean", "laundromat")),
    ("kesehatan", ("dokter", "klinik", "apotek", "dental", "hospital", "rumah sakit")),
)

_POPULAR_MERCHANTS = (
    "Alfamart", "Indomaret", "Alfamidi", "Shopee", "Tokopedia", "Lazada",
    "Mixue", "Mie Gacoan",
)

# â”€â”€ Singletons â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_mbanking_parser: Any = None
_receipt_parser: Any = None
_classifier: Any = None


def _get_mbanking_parser() -> Any:
    global _mbanking_parser
    if _mbanking_parser is None:
        from mbanking_inference import MBankingParser
        _mbanking_parser = MBankingParser()
    return _mbanking_parser


def _get_receipt_parser() -> Any:
    global _receipt_parser
    if _receipt_parser is None:
        from donut_inference import ReceiptParser
        _receipt_parser = ReceiptParser()
    return _receipt_parser


def _get_classifier() -> Any:
    global _classifier
    if _classifier is None:
        from indobert import HybridCategoryClassifier
        _classifier = HybridCategoryClassifier(MODEL_DIR)
    return _classifier


# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _normalize_donut_result(raw: Any) -> dict[str, Any]:
    """Safely coerce DONUT parse result to a plain dict.

    Handles every shape the DONUT wrapper may return:
      - dict  â†’ return as-is
      - list of dicts â†’ pick the best candidate (has "total" key)
      - empty list / None â†’ return {}
      - str â†’ wrap in {"raw_text": str}
      - anything else â†’ return {}
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        if not raw:
            return {}
        # Prefer the entry that contains "total" so amount can be extracted
        for item in raw:
            if isinstance(item, dict) and item.get("total"):
                return item
        # Fall back to first dict in list
        for item in raw:
            if isinstance(item, dict):
                return item
        return {}
    if isinstance(raw, str):
        return {"raw_text": raw}
    return {}


def _classify_category(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        return {"label": "lainnya", "confidence": 0.0}
    keyword_result = _keyword_category(text)
    if keyword_result:
        return keyword_result
    try:
        clf = _get_classifier()
        result = clf.predict(text)
        return {
            "label": result.get("label", "lainnya"),
            "confidence": float(result.get("confidence", 0.0)),
        }
    except Exception as exc:
        logger.warning("Category classifier failed: %s", exc)
        return {"label": "lainnya", "confidence": 0.0}


def _keyword_category(text: str) -> dict[str, Any] | None:
    haystack = str(text or "").lower()
    for label, keywords in _CATEGORY_KEYWORDS:
        for keyword in keywords:
            if len(keyword) <= 4:
                matched = bool(re.search(rf"\b{re.escape(keyword)}\b", haystack))
            else:
                matched = keyword in haystack
            if matched:
                return {"label": label, "confidence": 0.68}
    return None


def _category_from_merchant(merchant: str) -> dict[str, Any] | None:
    """Classify mostly from the final merchant; raw OCR is only fallback."""
    if not merchant or not merchant.strip():
        return None
    keyword_result = _keyword_category(merchant)
    if keyword_result:
        keyword_result["confidence"] = max(keyword_result["confidence"], 0.78)
        return keyword_result
    return None


def _normalize_amount(val: Any) -> float:
    try:
        v = float(val or 0)
        return max(0.0, v)
    except (TypeError, ValueError):
        return 0.0


def _safe_confidence(val: Any) -> float:
    try:
        return max(0.0, min(1.0, float(val or 0)))
    except (TypeError, ValueError):
        return 0.0


def _derive_status(
    *,
    merchant: str,
    amount: float,
    date: str,
    doc_type_conf: float,
    field_conf: dict[str, float],
) -> tuple[bool, str]:
    has_merchant = bool(merchant.strip())
    has_amount = amount > 0
    has_date = bool(date.strip())
    useful_fields = sum([has_merchant, has_amount, has_date])
    if useful_fields == 0:
        return False, "failed"

    merchant_ok = has_merchant and field_conf.get("merchant", 0.0) >= 0.55
    amount_ok = has_amount and field_conf.get("amount", 0.0) >= 0.55
    date_ok = has_date and field_conf.get("date", 0.0) >= 0.55
    doc_ok = doc_type_conf >= 0.62
    if merchant_ok and amount_ok and date_ok and doc_ok:
        return True, "extracted"
    return True, "needs_review"


def clean_merchant_candidate(text: str) -> str:
    """Clean a merchant candidate without relying on filenames or samples."""
    clean = re.sub(r"\s+", " ", str(text or "")).strip(" \t\r\n,.;:-!()[]{}")
    if not clean:
        return ""
    key = re.sub(r"[^a-z0-9]", "", clean.lower())
    for merchant_name in _POPULAR_MERCHANTS:
        merchant_key = re.sub(r"[^a-z0-9]", "", merchant_name.lower())
        if key == merchant_key:
            return merchant_name
        if len(key) >= 6 and SequenceMatcher(None, key, merchant_key).ratio() >= 0.88:
            return merchant_name
    hard_cut = _MERCHANT_HARD_CUT_RE.search(clean)
    if hard_cut and hard_cut.start() > 0:
        clean = clean[: hard_cut.start()].strip(" ,.;:-")
    addr = _MERCHANT_ADDR_RE.search(clean)
    if addr and addr.start() > 0:
        clean = clean[: addr.start()].strip(" ,.;:-")
    clean = re.sub(
        r"\s*,\s*(?:bandung|jakarta|bekasi|tangerang|surabaya|depok|bogor|semarang|"
        r"yogyakarta|jogja|medan|malang|denpasar|bali)\s*,?\s*id\b.*$",
        "",
        clean,
        flags=re.IGNORECASE,
    ).strip(" ,.;:-")
    if len(clean) > 60:
        shortened = re.sub(r"\s+\S*$", "", clean[:60]).strip(" ,.;:-")
        if len(shortened) >= 8:
            clean = shortened
    return clean


def is_valid_merchant_candidate(text: str) -> bool:
    """Reject labels, dates, IDs, amounts, and address-only text."""
    clean = clean_merchant_candidate(text)
    if not clean or len(clean) < 2:
        return False
    lower = clean.lower()
    if _MERCHANT_JUNK_RE.match(lower) or _MERCHANT_FORBIDDEN_RE.search(lower):
        return False
    if re.fullmatch(
        r"(?:bandung|jakarta|bekasi|tangerang|surabaya|depok|bogor|semarang|"
        r"yogyakarta|jogja|medan|malang|denpasar|bali)\s*,?\s*id",
        lower,
    ):
        return False
    if lower in {"xendit", "midtrans", "bank", "bank bca", "bank mandiri", "bank bni", "bank bri"}:
        return False
    if re.search(r"\b(rp|idr)\s*[\d.,oO]+\b", lower) or re.match(r"^[\-\s]*(rp|idr)\b", lower):
        return False
    if re.search(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b", clean):
        return False
    if re.search(r"\b\d{1,2}[:.]\d{2}(:\d{2})?\b", clean):
        return False
    if clean.isdigit():
        return False
    digits = re.sub(r"\D", "", clean)
    letters = re.sub(r"[^A-Za-z]", "", clean)
    if len(digits) >= 8 and len(digits) / max(len(clean), 1) > 0.45:
        return False
    if re.match(r"^[A-Z0-9]{6,}$", clean, re.IGNORECASE) and digits and letters:
        return False
    if re.match(r"^[A-Z0-9]{3,12}(?:-[A-Z0-9]{2,12}){1,4}$", clean, re.IGNORECASE):
        return False
    if len(letters) >= 6:
        vowels = sum(ch.lower() in "aeiou" for ch in letters)
        if vowels <= 1:
            return False
    return True


def choose_best_merchant_candidate(candidates: list[str], context: dict[str, Any] | None = None) -> str:
    """Pick the strongest validated merchant candidate."""
    scored: list[tuple[int, str]] = []
    for idx, raw in enumerate(candidates or []):
        clean = clean_merchant_candidate(raw)
        if not is_valid_merchant_candidate(clean):
            continue
        score = 100 - idx
        if 2 <= len(clean.split()) <= 5:
            score += 14
        if "," in clean:
            score += 6
        if len(clean) > 45:
            score -= 10
        scored.append((score, clean))
    if not scored:
        return ""
    return max(scored, key=lambda item: (item[0], len(item[1])))[1]


def _map_document_type(internal: str, fallback_route: str = "unknown") -> tuple[str, str, str]:
    internal_clean = (internal or "").strip().lower()
    if internal_clean in SCREENSHOT_INTERNAL_TYPES:
        public = "screenshot"
    elif internal_clean in RECEIPT_INTERNAL_TYPES:
        public = "receipt"
    elif fallback_route in ("receipt", "screenshot"):
        public = fallback_route
    else:
        public = "unknown"
    return public, DOC_TYPE_LABEL[public], internal_clean


def _detect_route_from_lines(ocr_lines: list[str]) -> tuple[str, float]:
    """Auto-detect document type from raw OCR lines.

    Returns (route_type, confidence):
      route_type: "screenshot" | "receipt"
      confidence: 0.0 â€“ 1.0
    """
    # First try the postprocessor's classifier (richer heuristics)
    try:
        from extraction_postprocessor import classify_document_type
        doc_type = classify_document_type(ocr_lines)
        _SC_TYPES = ("compact_qr_card", "mbanking_transaction_detail", "ewallet_receipt_screen")
        if doc_type in _SC_TYPES:
            return "screenshot", 0.90
        if doc_type == "photo_receipt":
            return "receipt", 0.85
    except Exception:
        pass

    # Fallback: lightweight keyword + layout scoring
    full = "\n".join(ocr_lines).lower()
    sc = sum(2 for kw in _SC_STRONG if kw in full)
    rc = sum(2 for kw in _RC_STRONG if kw in full)

    for line in ocr_lines:
        lower = line.lower()
        fixed = re.sub(r"[oO]", "0", lower)
        if re.search(r"\b(customer\s+pan|merchant\s+pan|terminal\s+id|reference|transaction\s+id|rrn|stan)\b", lower):
            sc += 3
        if re.search(r"\b(total|grand\s+total|subtotal|tunai|kembali|kembalian|kasir|ppn)\b", lower):
            rc += 2
        if re.search(r"\b(guest\s+folio|folio|balance\s+due|charges|credits|room|receipt\s+no)\b", lower):
            rc += 3
        if re.search(r"\b\d+\s*x\b|\b\d+x\b|\bx\d+\b|\bqty\b|\bpcs\b|@", lower):
            rc += 2
        if re.search(r"[a-z]{3,}.*(?:\d{1,3}[.,]\d{3}|\d+[.,]\d{2})\s*$", fixed):
            rc += 1

    if sc > rc:
        conf = min(0.92, 0.56 + (sc - rc) * 0.035)
        return "screenshot", conf
    if rc > sc:
        conf = min(0.92, 0.56 + (rc - sc) * 0.035)
        return "receipt", conf

    # When uncertain, receipt is the safer route unless strong payment infra won.
    # Status logic will still mark partial/weak results as needs_review.
    return "receipt", 0.45


# â”€â”€ Main entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def extract_from_image(image_path: Path, selected_type: str) -> dict[str, Any]:
    """Run OCR/parser pipeline and return a normalised extraction result dict.

    Args:
        image_path:    Path to the temporary uploaded image file.
        selected_type: "receipt" | "screenshot" | "auto"

    Returns:
        Dict matching ExtractionResponse schema (always valid JSON, never raises).
    """
    from extraction_postprocessor import postprocess

    warnings_list: list[str] = []
    debug_trace = ""
    merchant = ""
    amount = 0.0
    date = ""
    doc_type = "unknown"
    doc_type_conf = 0.0
    doc_type_source = "manual" if selected_type in ("receipt", "screenshot") else "auto"
    field_conf: dict[str, float] = {"merchant": 0.0, "amount": 0.0, "date": 0.0}
    raw_text = ""
    route = "unknown"
    success = True
    status = "extracted"
    started_at = time.perf_counter()
    stage_times: dict[str, float] = {}

    try:
        # â”€â”€ 1. Determine actual route â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        route_started = time.perf_counter()
        if selected_type == "auto":
            # Quick OCR scan for routing (reuse MBankingParser's EasyOCR)
            mb = _get_mbanking_parser()
            probe_lines = mb.extract_text_lines(str(image_path))
            route, doc_type_conf = _detect_route_from_lines(probe_lines)
            for key, value in getattr(mb, "last_timing", {}).items():
                if isinstance(value, (int, float)):
                    stage_times[f"route_ocr_{key}"] = float(value)
        else:
            route = selected_type          # "receipt" or "screenshot"
            doc_type_conf = 1.0            # user explicitly chose
            probe_lines = None
        stage_times["route"] = time.perf_counter() - route_started

        # â”€â”€ 2. Parse with the appropriate parser â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        parse_started = time.perf_counter()
        if route == "screenshot":
            raw = _extract_screenshot(image_path, postprocess, ocr_lines=probe_lines)
        else:
            raw = _extract_receipt(image_path, postprocess, ocr_lines=probe_lines)
        stage_times["parser_postprocess"] = time.perf_counter() - parse_started
        if probe_lines is None:
            mb = _get_mbanking_parser()
            for key, value in getattr(mb, "last_timing", {}).items():
                if isinstance(value, (int, float)):
                    stage_times[f"ocr_{key}"] = float(value)

        merchant_raw = raw.get("merchant", "") or ""
        merchant_candidates = list(raw.get("merchant_candidates", []) or [])
        merchant = choose_best_merchant_candidate([merchant_raw, *merchant_candidates], {"route": route})
        amount    = _normalize_amount(raw.get("amount", 0))
        date      = raw.get("date", "") or ""
        doc_type_internal = raw.get("document_type", route) or route
        if doc_type_source == "manual":
            doc_type = route if route in ("receipt", "screenshot") else "unknown"
            doc_type_label = DOC_TYPE_LABEL[doc_type]
            doc_type_internal = route
            doc_type_conf = 1.0
        else:
            doc_type, doc_type_label, doc_type_internal = _map_document_type(doc_type_internal, route)
        warnings_list = list(raw.get("warnings", []) or [])
        debug_trace   = str(raw.get("debug_trace", "") or "")
        timing_text = " ".join(
            f"{key}={int(value)}" if key.endswith("ocr_passes") else f"{key}={value:.2f}s"
            for key, value in stage_times.items()
        )
        debug_trace   = f"[TIMING] {timing_text}\n[ROUTE] selected_type={selected_type} route={route} confidence={doc_type_conf:.2f}\n{debug_trace}"
        raw_text      = str(raw.get("raw_text", "") or "")
        elapsed = time.perf_counter() - started_at
        if elapsed > MAX_EXTRACTION_SECONDS:
            warnings_list.append("Pembacaan membutuhkan waktu lebih lama. Periksa kembali hasilnya.")
            debug_trace = f"[TIMING] elapsed={elapsed:.2f}s soft_guard={MAX_EXTRACTION_SECONDS:.0f}s\n{debug_trace}"

        raw_conf  = raw.get("field_confidence", {}) or {}
        field_conf = {
            "merchant": _safe_confidence(raw_conf.get("merchant", 0)),
            "amount":   _safe_confidence(raw_conf.get("amount",   0)),
            "date":     _safe_confidence(raw_conf.get("date",     0)),
        }

        # â”€â”€ 3. Validate extracted fields â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if not merchant or merchant.upper() in ("MERCHANT TIDAK TERDETEKSI", ""):
            merchant = ""
            field_conf["merchant"] = 0.0
            warnings_list.append("Nama toko / penerima belum terbaca â€” silakan isi manual.")

        if amount <= 0:
            warnings_list.append("Nominal belum terbaca â€” silakan isi manual.")


        if not date:
            warnings_list.append("Tanggal belum terbaca - silakan isi manual.")

        success, status = _derive_status(
            merchant=merchant,
            amount=amount,
            date=date,
            doc_type_conf=doc_type_conf,
            field_conf=field_conf,
        )
        if elapsed > MAX_EXTRACTION_SECONDS and status == "extracted":
            status = "needs_review"
        if status == "needs_review":
            warnings_list.append("Beberapa data perlu diperiksa kembali sebelum disimpan.")

    except Exception as exc:
        logger.exception("Extraction pipeline failed: %s", exc)
        warnings_list.append("Transaksi belum berhasil dibaca. Isi detail transaksi secara manual.")
        success = False
        status  = "failed"
        merchant = ""
        if selected_type in ("receipt", "screenshot"):
            doc_type = selected_type
            doc_type_label = DOC_TYPE_LABEL[doc_type]
            doc_type_internal = selected_type
            doc_type_conf = 1.0
            doc_type_source = "manual"
        else:
            doc_type, doc_type_label, doc_type_internal = _map_document_type("", "unknown")

    # â”€â”€ 4. Category classification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    elapsed_for_category = time.perf_counter() - started_at
    category_started = time.perf_counter()
    cat_result = _category_from_merchant(merchant)
    if not cat_result:
        # Raw OCR is only a low-priority fallback because screenshot text often
        # contains banks, payment gateways, and transfer labels that are not the merchant.
        if elapsed_for_category > MAX_EXTRACTION_SECONDS:
            cat_result = {"label": "lainnya", "confidence": 0.0}
        else:
            cat_result = _classify_category(merchant or raw_text or "")
        if not merchant and raw_text:
            cat_result["confidence"] = min(cat_result["confidence"], 0.42)
    category     = cat_result["label"]
    cat_conf     = cat_result["confidence"]
    stage_times["category"] = time.perf_counter() - category_started
    if debug_trace:
        debug_trace = (
            f"[TIMING_CATEGORY] category={stage_times['category']:.2f}s "
            f"total={time.perf_counter() - started_at:.2f}s\n{debug_trace}"
        )

    return {
        "merchant":               merchant,
        "amount":                 amount,
        "date":                   date,
        "category":               category,
        "category_display":       CATEGORY_DISPLAY.get(category, "Lainnya"),
        "document_type":          doc_type,
        "document_type_label":    doc_type_label,
        "document_type_internal": doc_type_internal,
        "document_type_confidence": doc_type_conf,
        "document_type_source":   doc_type_source,
        "confidence": {
            "merchant": field_conf.get("merchant", 0.0),
            "amount":   field_conf.get("amount",   0.0),
            "date":     field_conf.get("date",     0.0),
            "category": cat_conf,
        },
        "warnings":     warnings_list,
        "success":      success,
        "status":       status,
        "debug_trace":  debug_trace,
    }


# â”€â”€ Route-specific extractors â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _extract_screenshot(image_path: Path, postprocess: Any, ocr_lines: list[str] | None = None) -> dict[str, Any]:
    """Extract from a payment/banking screenshot using MBankingParser."""
    parser = _get_mbanking_parser()
    parser_started = time.perf_counter()
    raw = parser.parse(str(image_path), pre_ocr_lines=ocr_lines)
    parser_elapsed = time.perf_counter() - parser_started

    ocr_lines         = raw.get("ocr_lines", []) or []
    existing_merchant = str(raw.get("recipient", "") or "")
    existing_amount   = _normalize_amount(raw.get("amount", 0))
    existing_date     = str(raw.get("date", "") or "")
    doc_type          = str(raw.get("screenshot_category", "mbanking_transaction_detail") or "mbanking_transaction_detail")

    post_started = time.perf_counter()
    post = postprocess(
        ocr_lines=ocr_lines,
        route_type="screenshot",
        existing_merchant=existing_merchant,
        existing_amount=existing_amount,
        existing_date=existing_date,
    )
    post_elapsed = time.perf_counter() - post_started
    debug_trace = (
        f"[TIMING_EXTRACTOR] parser={parser_elapsed:.2f}s postprocess={post_elapsed:.2f}s\n"
        f"{post.get('debug_trace', '')}"
    )

    return {
        "merchant":         post.get("merchant", existing_merchant) or existing_merchant,
        "merchant_candidates": [post.get("merchant", ""), existing_merchant],
        "amount":           _normalize_amount(post.get("amount", existing_amount)),
        "date":             post.get("date", existing_date) or existing_date,
        "document_type":    doc_type,
        "field_confidence": post.get("field_confidence", {}),
        "warnings":         post.get("warnings", []),
        "debug_trace":      debug_trace,
        "raw_text":         "\n".join(ocr_lines),
    }


def _extract_receipt(image_path: Path, postprocess: Any, ocr_lines: list[str] | None = None) -> dict[str, Any]:
    """Extract from a physical receipt photo using OCR, with DONUT only as fallback."""
    # Reuse MBankingParser's EasyOCR for raw text lines
    mb_parser  = _get_mbanking_parser()
    ocr_started = time.perf_counter()
    ocr_lines  = ocr_lines if ocr_lines is not None else mb_parser.extract_text_lines(str(image_path))
    ocr_elapsed = time.perf_counter() - ocr_started

    post_started = time.perf_counter()
    post = postprocess(
        ocr_lines=ocr_lines,
        route_type="receipt",
        donut_result=None,
    )
    post_elapsed = time.perf_counter() - post_started

    # DONUT is much slower on CPU. Demo default keeps it disabled unless this
    # flag is explicitly enabled for diagnostic comparison.
    donut_used = False
    if ENABLE_DONUT_FALLBACK and _normalize_amount(post.get("amount", 0)) <= 0:
        donut_raw: Any = {}
        try:
            receipt_parser = _get_receipt_parser()
            donut_raw = receipt_parser.parse(str(image_path))
        except Exception as exc:
            logger.warning("DONUT parse failed, OCR-only fallback: %s", exc)

        donut_result = _normalize_donut_result(donut_raw)
        if donut_result:
            donut_used = True
            post = postprocess(
                ocr_lines=ocr_lines,
                route_type="receipt",
                donut_result=donut_result,
            )

    debug_trace = post.get("debug_trace", "")
    debug_trace = f"[TIMING_EXTRACTOR] ocr={ocr_elapsed:.2f}s postprocess={post_elapsed:.2f}s\n{debug_trace}"
    if not ENABLE_DONUT_FALLBACK:
        debug_trace = f"[RECEIPT] donut_disabled=demo_fast_path\n{debug_trace}"
    elif not donut_used:
        debug_trace = f"[RECEIPT] donut_skipped=ocr_amount_found_or_not_needed\n{debug_trace}"
    else:
        debug_trace = f"[RECEIPT] donut_used=fallback_amount_missing\n{debug_trace}"

    return {
        "merchant":         post.get("merchant", "") or "",
        "merchant_candidates": [post.get("merchant", "")],
        "amount":           _normalize_amount(post.get("amount", 0)),
        "date":             post.get("date", "") or "",
        "document_type":    "receipt",
        "field_confidence": post.get("field_confidence", {}),
        "warnings":         post.get("warnings", []),
        "debug_trace":      debug_trace,
        "raw_text":         "\n".join(ocr_lines),
    }
