"""Sanity check pretrained DONUT pada 5 sample dari test set.

Ambil 1 dari kaggle, 2 dari cord, 2 dari mbanking → jalankan inference,
bandingkan dengan ground truth.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from donut_inference import ReceiptParser


def load_jsonl(path: Path) -> list[dict]:
    """Load file JSONL → list of dict."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def pick_samples(entries: list[dict]) -> list[dict]:
    """Pilih sample bervariasi dari 3 source untuk diuji."""
    by_source: dict[str, list[dict]] = {}
    for e in entries:
        by_source.setdefault(e["source"], []).append(e)
    return (
        by_source.get("kaggle",   [])[:1] +
        by_source.get("cord",     [])[:2] +
        by_source.get("mbanking", [])[:2]
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    PROJECT_ROOT = Path(r"C:\Users\USER\Documents\CAPSTONE")
    test_jsonl = PROJECT_ROOT / "data_processed" / "unified" / "test.jsonl"

    print("Loading test entries…")
    samples = pick_samples(load_jsonl(test_jsonl))
    print(f"Picked {len(samples)} samples\n")

    print("Initializing DONUT (run pertama download ~700 MB, ~2-5 menit)…")
    parser = ReceiptParser()
    print("Ready.\n")

    for i, entry in enumerate(samples, 1):
        print("=" * 72)
        print(f"[{i}/{len(samples)}] [{entry['source'].upper()}] "
              f"{Path(entry['image_path']).name}")

        gt = entry["donut_target"]
        print("  GROUND TRUTH:")
        print(f"    {gt[:200]}{'…' if len(gt) > 200 else ''}")

        try:
            t0 = time.time()
            result = parser.parse(entry["image_path"])
            elapsed = time.time() - t0
            print(f"\n  PREDICTION ({elapsed:.1f}s on CPU):")
            pretty = json.dumps(result, indent=2, ensure_ascii=False)
            # Indent untuk readability
            print("    " + pretty.replace("\n", "\n    "))
        except Exception as exc:                          # noqa: BLE001
            print(f"\n  ERROR: {type(exc).__name__}: {exc}")
        print()


if __name__ == "__main__":
    main()
    