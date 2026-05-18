"""Sample 10 record per kategori dari train_labeled.jsonl untuk verifikasi.

Output ini membantu spot keyword apa yang masih missing — fokus utama
ke kategori 'lainnya' yang harusnya sudah terkecil.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    PROJECT_ROOT = Path(r"C:\Users\USER\Documents\CAPSTONE")
    LABELED = PROJECT_ROOT / "data_processed" / "unified" / "train_labeled.jsonl"
    N_PER_LABEL = 10

    by_label: dict[str, list[dict]] = defaultdict(list)
    with open(LABELED, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_label[r["category_label"]].append(r)

    rng = random.Random(42)
    # Urut dari kategori paling banyak
    for label, records in sorted(by_label.items(), key=lambda kv: -len(kv[1])):
        sampled = rng.sample(records, min(N_PER_LABEL, len(records)))
        print(f"\n{'='*72}")
        print(f"[{label}]  total={len(records)}  showing {len(sampled)}")
        print("=" * 72)
        for r in sampled:
            text = (
                r.get("merchant")
                or r.get("text_for_classification")
                or "<empty>"
            )
            text = text.replace("\n", " ").strip()
            if len(text) > 100:
                text = text[:97] + "..."
            print(f"  [{r['source']:8s}] {text}")


if __name__ == "__main__":
    main()