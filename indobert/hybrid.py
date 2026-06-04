"""Hybrid classifier — keyword 3 tier + IndoBERT.

Tier 1: word-boundary keyword (dari auto_labeler.label_text)
Tier 2: substring (untuk compound: 'Abyfood' → food → Makanan)
Tier 3: kata-makanan-Indonesia word-boundary (untuk: 'Sop Burtok' → sop → Makanan,
        tapi NOT 'Sopir' karena pakai \\b)
Tier 4: IndoBERT fallback
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .auto_labeler import label_text
from .inference import CategoryClassifier


_SUBSTRING_PATTERNS: dict[str, list[str]] = {
    "makanan_minuman": [
        # Apps / platforms
        "shopeefood", "shopee food", "gofood", "grabfood", "goeat",
        # Venue types
        "food", "cafe", "kafe", "kopi", "warkop", "warung", "warteg",
        "resto", "restoran", "restaurant", "bakery", "bakeri", "depot", "dapur",
        "donut", "pizza", "burger", "sushi", "ramen", "dimsum",
        "fried chicken", "ayam goreng", "ayam bakar",
        "takoyaki", "hotdog", "kebab", "shawarma",
        "waffle", "pancake", "crepe", "boba", "bubble tea",
        # Known fast-food chains
        "mcdonald", "mc donald", " kfc ", "kentucky", "pizza hut",
        "burger king", "jco ", "j.co", "chatime", "starbuck", "breadtalk",
        # Drinks & brands
        "bread", "roti", "cream", "tea", "teh", "milk", "susu", "jus", "juice",
        "frestea", "minerale", "aice", "cimory", "aqua", "pocari",
        "mochi", "gelato", "es krim", "ice cream",
        # Generic food words
        "kitchen", "catering", "eatery", "dining", "makan",
        "rm.", "r.m.", "rm ", "waroenk",
    ],
    "kesehatan": [
        "klinik", "clinic", "dental", "apotek", "apotik",
        "pharmacy", "medical", "hospital", "rs ", "puskesmas",
        "laboratorium", "lab ", "optik", "optic", "dokter",
        "rumah sakit", "poli ", "fisioterapi",
    ],
    "pendidikan": [
        "bimbel", "kursus", "sekolah", "school",
        "kampus", "universitas", "university", "academy",
        "les ", "privat", "tutor", "edukasi",
    ],
    "hiburan": [
        "hotel", "resort", "villa", "cinema", "karaoke", "bioskop",
        "wisata", "museum", "taman ", "wahana", "game", "esport",
        "gym", "fitness", "spa", "salon", "laundry", "cuci",
    ],
    "belanja": [
        "mart", "store", "shop", "minimart", "supermart", "supermarket",
        "alfamart", "indomaret", "alfamidi", "hypermart", "carrefour", "giant",
        "lottemart", "transmart", "shopee", "tokopedia", "lazada",
        "blibli", "bukalapak", "toko", "swalayan",
    ],
    "transportasi": [
        "rental", "taksi", "taxi", "travel", "gojek", "grab",
        "spbu", "pertamina", "shell", "vivo", "tol", "parkir",
        "bus ", "kereta", "pesawat", "tiket", "bbm ", "bensin",
        "blue bird", "maxim",
    ],
    "tagihan": [
        "wifi", "pulsa", "listrik", "pln", "token", "air ", "pdam",
        "internet", "indihome", "firstmedia", "myrepublic",
        "bpjs", "asuransi", "insurance", "cicilan", "angsuran",
        "telepon", "phone", "gopay bills", "gobills", "tagihan",
        "telkom", "xl ", "telkomsel", "simpati", "by.u",
    ],
}

# Kata pendek yang HARUS pakai word-boundary (case insensitive) supaya tidak
# false-match "Sopir", "Sophie", "Aymara" dst.
_WORD_PATTERNS: dict[str, list[str]] = {
    "makanan_minuman": [
        "sop", "soto", "bakso", "mie", "mi", "nasi", "ayam",
        "ketoprak", "gado-gado", "gado gado", "rumah makan",
        "bubur", "rendang", "padang", "sate", "iga", "konro",
        "burtok", "lalapan", "pecel", "rawon", "siomay", "batagor",
        "kwetiau", "pempek", "mpek", "coto", "kantin",
        "lotek", "cendol", "dawet", "es teh", "kue", "jajan",
        "nasi goreng", "nasi bakar", "nasi uduk", "nasi box",
        "ayam kampus", "ayam geprek", "mie ayam", "bakmi",
        "gorengan", "cilok", "cireng", "batagor", "martabak",
        "es campur", "es buah", "jus buah",
        "coffee", "matcha", "cha", "tes",
    ],
    "tagihan": [
        "pln", "token",
    ],
    "transportasi": [
        "tol", "spbu",
    ],
    "belanja": [
        "mart",
    ],
}

# Pre-compile regex word-boundary patterns
_WORD_REGEX = {
    cat: [re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE) for w in words]
    for cat, words in _WORD_PATTERNS.items()
}


def _check_substring(text: str) -> dict[str, int]:
    text_lower = text.lower()
    counts: dict[str, int] = {}
    for cat, keywords in _SUBSTRING_PATTERNS.items():
        for kw in keywords:
            if kw in text_lower:
                counts[cat] = counts.get(cat, 0) + 1
    return counts


def _check_word_boundary(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cat, patterns in _WORD_REGEX.items():
        for pat in patterns:
            n = len(pat.findall(text))
            if n > 0:
                counts[cat] = counts.get(cat, 0) + n
    return counts


class HybridCategoryClassifier:
    """Keyword tiered + IndoBERT untuk 8 kategori."""

    def __init__(self, model_dir: str | Path) -> None:
        self.bert = CategoryClassifier(model_dir)

    def predict(self, text: str) -> dict[str, Any]:
        if not text or not text.strip():
            return {"label": "lainnya", "confidence": 0.0, "source": "empty"}

        # Run ALL keyword detectors
        _, kw_counts = label_text(text)
        substr_counts = _check_substring(text)
        word_counts = _check_word_boundary(text)

        # Merge counts
        combined: dict[str, int] = {}
        for source_dict in (kw_counts, substr_counts, word_counts):
            for cat, c in source_dict.items():
                if cat == "lainnya":
                    continue
                combined[cat] = combined.get(cat, 0) + c

        if combined:
            best_cat = max(combined.keys(), key=lambda k: combined[k])
            return {
                "label":       best_cat,
                "confidence":  0.90,
                "source":      "keyword",
                "match_count": combined[best_cat],
            }

        # Fallback IndoBERT
        result = self.bert.predict(text)
        result["source"] = "indobert"
        return result