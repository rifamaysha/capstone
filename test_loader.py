"""Smoke test: pastikan 3 loader bisa parse data dengan benar."""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from loaders import load_cord, load_kaggle, load_mbanking


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    PROJECT_ROOT = Path(r"C:\Users\USER\Documents\CAPSTONE")
    PROCESSED = PROJECT_ROOT / "data_processed"

    # ---------- Kaggle ----------
    print("\n" + "=" * 72 + "\nKAGGLE\n" + "=" * 72)
    kaggle = load_kaggle(
        jsonl_path=PROJECT_ROOT / "dataset" / "dataset_kaggle" / "donut_train.jsonl",
        image_dir=PROCESSED / "kaggle",
        csv_path=PROJECT_ROOT / "dataset" / "dataset_kaggle" / "train_list.csv",
    )
    print(f"  Total records   : {len(kaggle)}")
    cats = Counter(r.category for r in kaggle if r.category)
    print(f"  Top categories  : {dict(cats.most_common(8))}")
    if kaggle:
        s = kaggle[0]
        print(f"  Sample[0]       : {s.image_path.name}")
        print(f"    category={s.category!r}  year={s.transaction_date!r}")
        print(f"    text='{(s.text_for_classification or '')[:80]}...'")

    # ---------- CORD ----------
    print("\n" + "=" * 72 + "\nCORD\n" + "=" * 72)
    cord = load_cord(
        jsonl_path=PROJECT_ROOT / "dataset" / "dataset_hf" / "donut_train.jsonl",
        image_dir=PROCESSED / "huggingface",
    )
    print(f"  Total records   : {len(cord)}")
    if cord:
        avg_items = sum(len(r.items) for r in cord) / len(cord)
        with_total = sum(1 for r in cord if r.total_amount)
        print(f"  Avg items/struk : {avg_items:.1f}")
        print(f"  Has total_amount: {with_total}/{len(cord)}")
        s = cord[0]
        print(f"  Sample[0]       : {s.image_path.name}")
        print(f"    total={s.total_amount} IDR  items={len(s.items)}")
        if s.items:
            print(f"    first item: {s.items[0]}")

    # ---------- M-Banking ----------
    print("\n" + "=" * 72 + "\nM-BANKING\n" + "=" * 72)
    mbanking = load_mbanking(
        jsonl_path=PROJECT_ROOT / "dataset" / "dataset_mbanking" / "donut_train.jsonl",
        image_dir=PROCESSED / "mbanking",
    )
    print(f"  Total records   : {len(mbanking)}")
    banks = Counter(r.raw_metadata.get("bank") for r in mbanking)
    print(f"  Bank breakdown  : {dict(banks)}")
    if mbanking:
        s = mbanking[0]
        print(f"  Sample[0]       : {s.image_path.name}")
        print(f"    recipient='{(s.merchant or '')[:60]}...'")
        print(f"    amount={s.total_amount}  date={s.transaction_date!r}")

    # ---------- Total ----------
    total = len(kaggle) + len(cord) + len(mbanking)
    print("\n" + "=" * 72)
    print(f"GRAND TOTAL: {total} records "
          f"(kaggle={len(kaggle)}, cord={len(cord)}, mbanking={len(mbanking)})")
    print("=" * 72)


if __name__ == "__main__":
    main()