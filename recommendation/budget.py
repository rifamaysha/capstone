from __future__ import annotations

from typing import Any

# Bucket mapping 
# Maps internal category keys → budget bucket
CATEGORY_BUCKET: dict[str, str] = {
    "makanan_minuman": "kebutuhan",
    "transportasi":    "kebutuhan",
    "tagihan":         "kebutuhan",
    "kesehatan":       "kebutuhan",
    "pendidikan":      "kebutuhan",
    "belanja":         "keinginan",
    "hiburan":         "keinginan",
    "lainnya":         "keinginan",
}

BUCKET_DISPLAY: dict[str, str] = {
    "kebutuhan": "Kebutuhan",
    "keinginan": "Keinginan",
    "tabungan":  "Tabungan / Investasi",
}

# Ideal 50/30/20 ratio
IDEAL_RATIO: dict[str, float] = {
    "kebutuhan": 0.50,
    "keinginan": 0.30,
    "tabungan":  0.20,
}


def analyze_budget(
    transactions: list[dict[str, Any]],
    monthly_income: float,
) -> dict[str, Any]:
    """Analyse spending against the 50/30/20 budget rule.

    Returns a dict with:
        bucket_totals   – actual Rp spent per bucket
        actual_ratio    – actual proportion per bucket
        ideal_ratio     – target proportion (50/30/20)
        tabungan        – estimated savings (income − total_spent)
        total_spent     – sum of all transaction amounts
        recommendations – list of Indonesian recommendation strings
    """
    if monthly_income <= 0:
        return {
            "bucket_totals": {},
            "actual_ratio": {},
            "ideal_ratio": IDEAL_RATIO,
            "tabungan": 0.0,
            "total_spent": 0.0,
            "recommendations": [],
        }

    # Sum per bucket
    bucket_totals: dict[str, float] = {"kebutuhan": 0.0, "keinginan": 0.0}
    for tx in transactions:
        amt = float(tx.get("amount") or 0)
        if amt <= 0:
            continue
        cat = str(tx.get("category") or "lainnya")
        bucket = CATEGORY_BUCKET.get(cat, "keinginan")
        bucket_totals[bucket] = bucket_totals.get(bucket, 0.0) + amt

    total_spent = sum(bucket_totals.values())
    tabungan = max(0.0, monthly_income - total_spent)

    # Actual ratios
    actual_ratio: dict[str, float] = {}
    for bucket, total in bucket_totals.items():
        actual_ratio[bucket] = round(total / monthly_income, 4)
    actual_ratio["tabungan"] = round(tabungan / monthly_income, 4)

    # Build recommendations 
    recs: list[str] = []

    kebutuhan_pct  = actual_ratio.get("kebutuhan", 0) * 100
    keinginan_pct  = actual_ratio.get("keinginan", 0) * 100
    tabungan_pct   = actual_ratio.get("tabungan",  0) * 100
    total_pct      = (total_spent / monthly_income) * 100

    # Over budget overall
    if total_spent > monthly_income:
        over = total_spent - monthly_income
        recs.append(
            f"Pengeluaran melebihi pemasukan sebesar Rp {over:,.0f}. "
            "Coba kurangi pengeluaran keinginan terlebih dahulu."
        )

    # Kebutuhan > 50%
    if kebutuhan_pct > 55:
        recs.append(
            f"Pengeluaran kebutuhan ({kebutuhan_pct:.0f}%) melebihi batas ideal 50%. "
            "Cek tagihan atau biaya transportasi yang bisa dihemat."
        )

    # Keinginan > 30%
    if keinginan_pct > 35:
        recs.append(
            f"Pengeluaran keinginan ({keinginan_pct:.0f}%) melebihi batas ideal 30%. "
            "Pertimbangkan untuk mengurangi belanja atau hiburan."
        )

    # Tabungan rendah
    if tabungan_pct < 10 and total_spent < monthly_income:
        recs.append(
            f"Tabungan saat ini hanya {tabungan_pct:.0f}% dari pemasukan. "
            "Idealnya sisihkan minimal 20% untuk tabungan atau investasi."
        )
    elif tabungan_pct >= 20:
        recs.append(
            f"Bagus! Kamu berhasil menyisihkan {tabungan_pct:.0f}% pemasukan untuk tabungan. "
            "Pertahankan kebiasaan ini."
        )

    # No recs yet → generic positive
    if not recs:
        recs.append(
            f"Pengeluaran kamu terkontrol dengan baik ({total_pct:.0f}% dari pemasukan). "
            "Tetap pantau setiap kategori agar tetap sesuai rencana."
        )

    return {
        "bucket_totals": bucket_totals,
        "actual_ratio":  actual_ratio,
        "ideal_ratio":   IDEAL_RATIO,
        "tabungan":      round(tabungan, 2),
        "total_spent":   round(total_spent, 2),
        "recommendations": recs,
    }
