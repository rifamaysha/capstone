"""End-to-end pipeline test — gambar → ekstraksi → klasifikasi → record.

Memvalidasi semua komponen Smart Personal Expense bekerja bersama:
preprocessing (sudah dilakukan), DONUT/M-Banking router, dan
HybridCategoryClassifier — menghasilkan transaksi terstruktur lengkap.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from donut_inference import ReceiptParser
from mbanking_inference import MBankingParser
from indobert import CATEGORY_DISPLAY, HybridCategoryClassifier


def is_screenshot_path(image_path: Path) -> bool:
    """Routing heuristic: gambar dari folder 'mbanking' = screenshot.

    Di Streamlit nanti user akan pilih sendiri via radio button — di sini
    kita pakai struktur folder untuk demo otomatis.
    """
    return "mbanking" in str(image_path).lower()


def _parse_total_to_int(s: Any) -> float | None:
    """Convert string total ('72,500' / '1.591.600') ke float."""
    if not s:
        return None
    cleaned = "".join(c for c in str(s) if c.isdigit())
    return float(cleaned) if cleaned else None


def process_end_to_end(
    image_path: Path,
    donut: ReceiptParser,
    mbanking: MBankingParser,
    classifier: HybridCategoryClassifier,
) -> dict[str, Any]:
    """Proses satu gambar → record transaksi lengkap."""
    is_screenshot = is_screenshot_path(image_path)
    t0 = time.time()

    # ---- ROUTING + EXTRACTION ----
    if is_screenshot:
        ext = mbanking.parse(image_path, return_raw=True)   # CHANGED: return_raw
        merchant = ext.get("recipient")
        amount   = ext.get("amount")
        date     = ext.get("date")
        items: list[str] = []
        # Classification text: merchant kalau ada, fallback ke OCR raw text
        text_for_class = merchant or (ext.get("raw_text") or "")[:200]   # CHANGED
        method = "easyocr+regex"
    else:
        ext = donut.parse(image_path)
        menu = ext.get("menu", [])
        if isinstance(menu, dict):
            menu = [menu]
        items = [m.get("nm", "") for m in menu
                 if isinstance(m, dict) and m.get("nm")]
        total_data = ext.get("total", {})
        amount = _parse_total_to_int(
            total_data.get("total_price") if isinstance(total_data, dict) else None
        )
        merchant = None
        date = None
        text_for_class = " ".join(items)
        method = "donut"

    # ---- CLASSIFICATION ----
    if text_for_class.strip():
        cls = classifier.predict(text_for_class)
    else:
        cls = {"label": "lainnya", "confidence": 0.0, "source": "empty"}

    return {
        "image":         image_path.name,
        "method":        method,
        "merchant":      merchant,
        "amount":        amount,
        "date":          date,
        "items_preview": items[:3],
        "n_items":       len(items),
        "category":      CATEGORY_DISPLAY.get(cls["label"], cls["label"]),
        "confidence":    cls["confidence"],
        "source":        cls.get("source", "?"),
        "elapsed_s":     round(time.time() - t0, 1),
    }


def print_record(rec: dict[str, Any]) -> None:
    """Tampilkan record dalam format readable."""
    print(f"  Image      : {rec['image']}")
    print(f"  Method     : {rec['method']:<14s} ({rec['elapsed_s']}s)")
    print(f"  Merchant   : {rec['merchant'] or '—'}")
    if rec["amount"] is not None:
        print(f"  Amount     : Rp {rec['amount']:,.0f}")
    else:
        print(f"  Amount     : —")
    print(f"  Date       : {rec['date'] or '—'}")
    if rec["items_preview"]:
        more = "…" if rec["n_items"] > 3 else ""
        print(f"  Items      : {', '.join(rec['items_preview'])}{more}  "
              f"({rec['n_items']} total)")
    print(f"  Category   : {rec['category']}  "
          f"({rec['confidence']:.0%} via {rec['source']})")


def main() -> None:
    # Quiet down library logs
    logging.basicConfig(level=logging.WARNING)
    for name in ["transformers", "huggingface_hub", "httpx", "easyocr"]:
        logging.getLogger(name).setLevel(logging.ERROR)

    PROJECT_ROOT = Path(r"C:\Users\USER\Documents\CAPSTONE")
    PROCESSED   = PROJECT_ROOT / "data_processed"
    MODEL_DIR   = PROJECT_ROOT / "models" / "indobert" / "run1" / "final"

    # Pilih sample dari masing-masing source
    samples: list[tuple[str, Path]] = []
    for label, root, n in [
        ("CORD",      PROCESSED / "huggingface", 1),
        ("M-Banking", PROCESSED / "mbanking",    2),
        ("Kaggle",    PROCESSED / "kaggle",      1),
    ]:
        imgs = sorted(root.rglob("*.png"))
        for img in imgs[:n]:
            samples.append((label, img))

    # ---- LOAD MODELS ----
    print("=" * 72)
    print("LOADING MODELS  (one-time setup, ~30 detik)")
    print("=" * 72)
    print("[1/3] DONUT pretrained...")
    donut = ReceiptParser()
    print("[2/3] EasyOCR (Indonesian + English)...")
    mbanking = MBankingParser()
    print("[3/3] IndoBERT hybrid classifier...")
    classifier = HybridCategoryClassifier(MODEL_DIR)
    print("Ready.\n")

    # ---- PROCESS ----
    print("=" * 72)
    print(f"END-TO-END PIPELINE TEST  ·  {len(samples)} samples")
    print("=" * 72)

    success = 0
    for i, (label, img_path) in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}]  [{label}]")
        print("-" * 72)
        try:
            rec = process_end_to_end(img_path, donut, mbanking, classifier)
            print_record(rec)
            success += 1
        except Exception as exc:                               # noqa: BLE001
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 72)
    print(f"COMPLETE  ·  {success}/{len(samples)} processed without errors")
    print("=" * 72)


if __name__ == "__main__":
    main()