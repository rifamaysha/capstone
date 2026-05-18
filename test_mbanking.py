"""Test parser M-Banking di sample dari test set."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from mbanking_inference import MBankingParser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    PROJECT_ROOT = Path(r"C:\Users\USER\Documents\CAPSTONE")
    test_jsonl = PROJECT_ROOT / "data_processed" / "unified" / "test.jsonl"

    # Ambil hanya entry mbanking
    with open(test_jsonl, encoding="utf-8") as f:
        mbanking_entries = [
            json.loads(line) for line in f
            if line.strip() and json.loads(line).get("source") == "mbanking"
        ][:5]   # 5 sampel saja

    print(f"Picked {len(mbanking_entries)} M-Banking samples\n")

    print("Initializing EasyOCR (run pertama download ~64 MB models)…")
    parser = MBankingParser()
    print("Ready.\n")

    for i, entry in enumerate(mbanking_entries, 1):
        print("=" * 72)
        print(f"[{i}/{len(mbanking_entries)}] {Path(entry['image_path']).name}")
        print(f"  GROUND TRUTH:")
        print(f"    amount    = {entry.get('total_amount')}")
        print(f"    date      = {entry.get('transaction_date')!r}")
        print(f"    recipient = {entry.get('merchant')!r}")

        try:
            t0 = time.time()
            result = parser.parse(entry["image_path"], return_raw=True)
            elapsed = time.time() - t0
            print(f"\n  PREDICTION ({elapsed:.1f}s on CPU):")
            print(f"    amount    = {result['amount']}")
            print(f"    date      = {result['date']!r}")
            print(f"    recipient = {result['recipient']!r}")
            print(f"    n_lines   = {result['n_lines']} OCR lines extracted")
            print(f"\n  RAW OCR (preview):")
            preview = result.get("raw_text", "")[:500]
            print(f"    {preview}{'…' if len(preview) >= 500 else ''}")
        except Exception as exc:                    # noqa: BLE001
            print(f"\n  ERROR: {type(exc).__name__}: {exc}")
        print()


if __name__ == "__main__":
    main()