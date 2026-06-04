"""Test hybrid classifier (keyword + IndoBERT) di berbagai input."""
from __future__ import annotations

from pathlib import Path

from indobert import CATEGORY_DISPLAY, HybridCategoryClassifier


def main() -> None:
    model_dir = Path(r"C:\Users\USER\Documents\CAPSTONE\models\indobert\run1\final")

    print("Loading hybrid classifier (keyword + IndoBERT)...")
    clf = HybridCategoryClassifier(model_dir)
    print(f"Loaded.\n")

    test_inputs = [
        # --- Kelas terlatih IndoBERT ---
        "Warmindo RF, Jl Terusan Buah Batu",        # makanan
        "GoFood Nasi Padang Sederhana",              # makanan
        "Starbucks Coffee Latte Croissant",          # makanan
        "tukang ayam 5",                             # makanan
        "SHELL PELAJAR PEJUANG-1",                   # transportasi
        "Gojek GoRide ke kampus",                    # transportasi
        "Indomaret Bojongsoang",                     # belanja
        "Tokopedia PT Belanja Online",               # belanja
        "Cinepolis Tiket Bioskop XXI",               # hiburan
        "Hotel Santika Bandung",                     # hiburan
        # --- Kelas tipis (hanya bisa via keyword) ---
        "FAMILY DENTAL CARE BUAH B",                 # kesehatan
        "Apotek Kimia Farma",                        # kesehatan
        "172 - GRAMEDIA BUAH BATU",                  # pendidikan
        "Bimbel Ganesha Operation",                  # pendidikan
        "PLN Listrik Token",                         # tagihan
        "Telkomsel Pulsa Paket Data",                # tagihan
        # --- Edge cases ---
        "Transfer ke teman",                         # lainnya
        "ABC123 XYZ TRANSFER",                       # benar-benar tidak jelas
        # --- Compound words (substring matching) ---
        "Abyfood 1",                                 # → makanan via "food"
        "Foodcourt Mall Bandung",                    # → makanan via "food"
        "Warkopku Senayan",                          # → makanan via "warkop"
        "MyBakery Donuts",                           # → makanan via "bakery"
        "Indomart Cilegon",                          # → belanja via "mart"
        "Klinik Apollo",                             # → kesehatan via "klinik"
        "Cinema XXI Plaza",                          # → hiburan via "cinema"
    ]

    header = f"{'INPUT':<48s}  {'PREDICTION':<22s}  {'CONF':>6s}  {'SOURCE':<10s}"
    print(header)
    print("-" * len(header))
    for text in test_inputs:
        r = clf.predict(text)
        display = CATEGORY_DISPLAY.get(r["label"], r["label"])
        print(f"{text[:46]:<48s}  {display:<22s}  "
              f"{r['confidence']:>6.1%}  {r.get('source', '?'):<10s}")


if __name__ == "__main__":
    main()