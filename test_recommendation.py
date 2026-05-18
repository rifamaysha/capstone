"""Test budget analyzer + anomaly detector dengan synthetic transaction history."""
from __future__ import annotations

from recommendation import (
    BUCKET_DISPLAY,
    IDEAL_RATIO,
    analyze_budget,
    detect_anomalies,
)
from indobert import CATEGORY_DISPLAY


def make_sample_history() -> list[dict]:
    """Sintetis 30 hari transaksi user, distribusi mirip pengeluaran kuliah."""
    return [
        # Makanan harian — banyak, kecil-kecil (~Rp 20-40k)
        {"date": "2026-04-01", "merchant": "Warmindo",            "amount":  18000, "category": "makanan_minuman"},
        {"date": "2026-04-02", "merchant": "GoFood Nasi Padang",  "amount":  35000, "category": "makanan_minuman"},
        {"date": "2026-04-03", "merchant": "Starbucks",           "amount":  48000, "category": "makanan_minuman"},
        {"date": "2026-04-04", "merchant": "Indomaret bekal",     "amount":  22000, "category": "makanan_minuman"},
        {"date": "2026-04-05", "merchant": "Ketoprak Bang Toyib", "amount":  16000, "category": "makanan_minuman"},
        {"date": "2026-04-06", "merchant": "Padang Sederhana",    "amount":  32000, "category": "makanan_minuman"},
        {"date": "2026-04-12", "merchant": "Hokben Mall",         "amount":  85000, "category": "makanan_minuman"},
        {"date": "2026-04-15", "merchant": "Steak Premium 21",    "amount": 350000, "category": "makanan_minuman"},  # outlier
        {"date": "2026-04-20", "merchant": "Gofood Mie Ayam",     "amount":  20000, "category": "makanan_minuman"},
        {"date": "2026-04-25", "merchant": "Warung Pak Slamet",   "amount":  19000, "category": "makanan_minuman"},
        # Transportasi
        {"date": "2026-04-01", "merchant": "Gojek GoRide",        "amount":  15000, "category": "transportasi"},
        {"date": "2026-04-08", "merchant": "Shell Pertamax",      "amount": 100000, "category": "transportasi"},
        {"date": "2026-04-15", "merchant": "Grab Car",            "amount":  45000, "category": "transportasi"},
        {"date": "2026-04-22", "merchant": "Gojek GoRide",        "amount":  12000, "category": "transportasi"},
        {"date": "2026-04-28", "merchant": "Shell Pertamax",      "amount":  95000, "category": "transportasi"},
        # Belanja
        {"date": "2026-04-10", "merchant": "Tokopedia case HP",   "amount":  85000, "category": "belanja"},
        {"date": "2026-04-18", "merchant": "Indomaret",           "amount":  50000, "category": "belanja"},
        # Hiburan — termasuk 1 outlier (staycation)
        {"date": "2026-04-13", "merchant": "Cinepolis Tiket",     "amount":  60000, "category": "hiburan"},
        {"date": "2026-04-21", "merchant": "Netflix subscription","amount": 120000, "category": "hiburan"},
        {"date": "2026-04-26", "merchant": "Hotel Santika",       "amount": 850000, "category": "hiburan"},  # outlier
        # Tagihan
        {"date": "2026-04-05", "merchant": "PLN Token",           "amount": 150000, "category": "tagihan"},
        {"date": "2026-04-05", "merchant": "Telkomsel paket data","amount": 100000, "category": "tagihan"},
        {"date": "2026-04-05", "merchant": "Indihome",            "amount": 350000, "category": "tagihan"},
        # Kesehatan
        {"date": "2026-04-14", "merchant": "Apotek Kimia Farma",  "amount":  85000, "category": "kesehatan"},
        # Pendidikan
        {"date": "2026-04-09", "merchant": "Gramedia buku",       "amount": 125000, "category": "pendidikan"},
    ]


def main() -> None:
    txs = make_sample_history()
    income = 5_000_000  # Rp 5 juta/bulan

    # ----------------- BUDGET ANALYSIS -----------------
    print("=" * 72)
    print("BUDGET ANALYSIS  ·  Aturan 50/30/20")
    print("=" * 72)
    result = analyze_budget(txs, income)
    print(f"  Pendapatan bulanan  : Rp {result['income']:>12,.0f}")
    print(f"  Total pengeluaran   : Rp {result['total_spent']:>12,.0f}")
    print(f"  Sisa (Tabungan)     : Rp {result['tabungan']:>12,.0f}")
    print()
    print(f"  {'BUCKET':<12s}  {'AKTUAL':>14s}  {'RASIO':>8s}  "
          f"{'IDEAL':>8s}  STATUS")
    print(f"  {'-'*12}  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*6}")
    for bucket in ["kebutuhan", "keinginan", "tabungan"]:
        amt = result["bucket_totals"].get(bucket, result["tabungan"])
        ratio = result["actual_ratio"][bucket]
        ideal = IDEAL_RATIO[bucket]
        status = "OK ✓" if abs(ratio - ideal) <= 0.05 else (
            "OVER ✗" if ratio > ideal else "UNDER"
        )
        print(f"  {BUCKET_DISPLAY[bucket]:<12s}  Rp {amt:>11,.0f}  "
              f"{ratio:>7.0%}   {ideal:>7.0%}   {status}")

    print()
    print("  Rekomendasi:")
    for rec in result["recommendations"]:
        print(f"    • {rec}")

    # ----------------- BREAKDOWN PER KATEGORI -----------------
    print()
    print("  Breakdown per kategori:")
    sorted_cats = sorted(
        result["by_category"].items(), key=lambda x: -x[1]
    )
    for cat, total in sorted_cats:
        cat_display = CATEGORY_DISPLAY.get(cat, cat)
        pct = total / result["total_spent"] if result["total_spent"] else 0
        print(f"    {cat_display:<22s}  Rp {total:>10,.0f}  ({pct:.0%})")

    # ----------------- ANOMALY DETECTION -----------------
    print()
    print("=" * 72)
    print("ANOMALY DETECTION  ·  Isolation Forest")
    print("=" * 72)
    anomalies = detect_anomalies(txs, contamination=0.1, by_category=True)
    if anomalies:
        print(f"  Ditemukan {len(anomalies)} transaksi outlier:")
        print()
        print(f"  {'MERCHANT':<28s}  {'AMOUNT':>12s}  "
              f"{'KATEGORI':<22s}  SCORE")
        print(f"  {'-'*28}  {'-'*12}  {'-'*22}  {'-'*6}")
        for a in sorted(anomalies, key=lambda x: x["anomaly_score"]):
            cat_display = CATEGORY_DISPLAY.get(a["category"], a["category"])
            print(f"  {a['merchant'][:26]:<28s}  Rp {a['amount']:>9,.0f}  "
                  f"{cat_display:<22s}  {a['anomaly_score']:>5.2f}")
    else:
        print("  Tidak ada anomali terdeteksi.")
    print()


if __name__ == "__main__":
    main()