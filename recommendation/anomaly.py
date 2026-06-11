"""Anomaly detection — Isolation Forest pada nilai transaksi.

Improvements over v1:
- Filters out invalid transactions (amount=0, missing category) before fitting
- Adjusts contamination dynamically based on sample size
- Falls back to a global model for categories that have too few samples
- Returns user-friendly anomaly_reason + baseline_median/mean/scope
- No crash on empty or tiny datasets

Inline self-test (runs on import only when __name__ == "__main__"):
  Normal: 25000, 27000, 30000, 28000, 26000
  Outlier: 250000
  Expected: 250000 flagged; normal transactions mostly not flagged.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

# Internal category key → human-readable Indonesian label
_CAT_DISPLAY: dict[str, str] = {
    "makanan_minuman": "Makanan & Minuman",
    "transportasi":    "Transportasi",
    "belanja":         "Belanja",
    "kesehatan":       "Kesehatan",
    "tagihan":         "Tagihan",
    "pendidikan":      "Pendidikan",
    "hiburan":         "Hiburan",
    "lainnya":         "Lainnya",
}


def _cat_label(cat: str) -> str:
    return _CAT_DISPLAY.get(cat, cat.replace("_", " ").title())


def _safe_contamination(n_samples: int, requested: float) -> float:
    """Return a contamination value that is safe for the given sample count.

    IsolationForest requires 0 < contamination < 0.5 and at least 2 outlier
    slots (floor(contamination * n) >= 1).  For tiny datasets we back off to
    a lower rate so we don't flag half the transactions as anomalies.
    """
    # At most 20% and at least 2%
    c = max(0.02, min(0.20, requested))
    # Ensure floor(c * n) >= 1 — i.e. c >= 1/n
    if n_samples >= 2:
        c = max(c, 1.0 / n_samples)
    # Never exceed 49%
    return min(c, 0.49)


def _detect_in_group(
    transactions: list[dict[str, Any]],
    contamination: float,
    random_state: int,
    scope: str = "category",
    baseline_transactions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Fit Isolation Forest on one group and return outliers with rich metadata."""
    if not transactions:
        return []

    amounts = np.array(
        [float(t.get("amount") or 0) for t in transactions]
    ).reshape(-1, 1)
    n = len(transactions)

    effective_contamination = _safe_contamination(n, contamination)

    try:
        model = IsolationForest(
            contamination=effective_contamination,
            random_state=random_state,
            n_estimators=100,
        )
        predictions = model.fit_predict(amounts)
        scores = model.score_samples(amounts)
    except Exception as exc:
        logger.warning("IsolationForest fit failed (scope=%s, n=%d): %s", scope, n, exc)
        return []

    # Baseline stats for the reason message
    baseline_txs = baseline_transactions or transactions
    baseline_amounts = np.array(
        [float(t.get("amount") or 0) for t in baseline_txs if float(t.get("amount") or 0) > 0]
    )
    if len(baseline_amounts) == 0:
        baseline_amounts = amounts.flatten()

    baseline_median = float(np.median(baseline_amounts))
    baseline_mean   = float(np.mean(baseline_amounts))

    # Category display name (extract from "category:xxx" scope format)
    cat_key = scope.split(":")[-1] if ":" in scope else scope
    cat_label = _cat_label(cat_key) if scope != "global_fallback" else "transaksi keseluruhan"

    outliers: list[dict[str, Any]] = []
    for tx, pred, score in zip(transactions, predictions, scores):
        if pred != -1:
            continue
        amt = float(tx.get("amount") or 0)
        tx_out = dict(tx)
        tx_out["anomaly_score"]    = float(score)
        tx_out["is_anomaly"]       = True
        tx_out["baseline_median"]  = round(baseline_median, 2)
        tx_out["baseline_mean"]    = round(baseline_mean, 2)
        tx_out["baseline_scope"]   = scope

        # User-friendly Indonesian reason
        if baseline_median > 0 and amt > baseline_median * 3:
            reason = (
                f"Nominal jauh lebih tinggi dari pola {cat_label} "
                f"(median Rp {baseline_median:,.0f})"
            )
        elif baseline_median > 0 and amt > 0 and amt < baseline_median / 3:
            reason = (
                f"Nominal jauh lebih rendah dari pola {cat_label} "
                f"(median Rp {baseline_median:,.0f})"
            )
        else:
            reason = (
                f"Nominal berbeda signifikan dari pola {cat_label} "
                f"(median Rp {baseline_median:,.0f})"
            )
        tx_out["anomaly_reason"] = reason
        outliers.append(tx_out)

    return outliers


def detect_anomalies(
    transactions: list[dict[str, Any]],
    contamination: float = 0.1,
    min_samples: int = 5,
    by_category: bool = True,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Deteksi transaksi outlier dengan Isolation Forest.

    Args:
        transactions: list of dict.  Each must have 'amount' (float) and
            optionally 'category'.  Transactions with amount <= 0 are ignored.
        contamination: estimated outlier proportion (0.02–0.20 clamped).
        min_samples: minimum transactions to fit a per-category model.
            Categories with fewer samples fall back to the global model.
        by_category: if True, fit per-category first; small categories use a
            global fallback instead of being silently skipped.
        random_state: reproducibility seed.

    Returns:
        List of outlier transaction dicts, each augmented with:
            - anomaly_score     (float, more negative = more anomalous)
            - is_anomaly        (True)
            - anomaly_reason    (Indonesian human-readable string)
            - baseline_median   (float)
            - baseline_mean     (float)
            - baseline_scope    ("category:xxx" or "global_fallback")
    """
    # ---- 1. Filter invalid transactions ----
    valid: list[dict[str, Any]] = [
        t for t in (transactions or [])
        if float(t.get("amount") or 0) > 0
    ]

    if len(valid) < 2:
        # ---- Rule-based fallback: flag if amount > 5x the single other tx ----
        if len(valid) == 1:
            return []
        return []

    # ---- 1b. Rule-based detection: always run regardless of sample size ----
    # Flag any transaction that is > 3x the mean of ALL transactions.
    # This catches obvious outliers (e.g. IBox 7jt when others are 34rb) even
    # when Isolation Forest can't fit due to small sample size.
    amounts = [float(t.get("amount") or 0) for t in valid]
    mean_amt = sum(amounts) / len(amounts)
    rule_flagged_ids: set = set()
    rule_anomalies: list[dict[str, Any]] = []

    if mean_amt > 0:
        for t in valid:
            amt = float(t.get("amount") or 0)
            if amt > mean_amt * 3:
                tx_out = dict(t)
                tx_out["anomaly_score"] = -0.6
                tx_out["is_anomaly"] = True
                tx_out["baseline_median"] = sorted(amounts)[len(amounts) // 2]
                tx_out["baseline_mean"] = round(mean_amt, 2)
                tx_out["baseline_scope"] = "rule_based"
                tx_out["anomaly_reason"] = (
                    f"Nominal Rp {amt:,.0f} jauh lebih tinggi dari rata-rata "
                    f"transaksi lain (Rp {mean_amt:,.0f})"
                )
                rule_flagged_ids.add(t.get("id"))
                rule_anomalies.append(tx_out)

    if not by_category:
        return _detect_in_group(valid, contamination, random_state, scope="global")

    # ---- 2. Split by category ----
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for tx in valid:
        cat = str(tx.get("category") or "lainnya")
        by_cat.setdefault(cat, []).append(tx)

    anomalies: list[dict[str, Any]] = []
    small_cat_txs: list[dict[str, Any]] = []

    for cat, txs in by_cat.items():
        if len(txs) >= min_samples:
            anomalies.extend(
                _detect_in_group(txs, contamination, random_state, scope=f"category:{cat}")
            )
        else:
            # Collect for global fallback below
            small_cat_txs.extend(txs)

    # ---- 3. Global fallback for under-represented categories ----
    if small_cat_txs:
        if len(small_cat_txs) >= 2:
            # Use all valid transactions as baseline so the outlier threshold is
            # calibrated to overall spending, not just the tiny category.
            anomalies.extend(
                _detect_in_group(
                    small_cat_txs,
                    contamination,
                    random_state,
                    scope="global_fallback",
                    baseline_transactions=valid,
                )
            )
        # If even the fallback group is < 2, skip silently (can't fit a model)

    # ---- 4. Merge: add rule-based anomalies not already caught by IsolationForest ----
    iso_ids = {a.get("id") for a in anomalies}
    for ra in rule_anomalies:
        if ra.get("id") not in iso_ids:
            anomalies.append(ra)

    return anomalies


# Quick inline self-test — only runs when script is executed directly
if __name__ == "__main__":
    _synthetic = [
        {"amount": 25000, "category": "makanan_minuman"},
        {"amount": 27000, "category": "makanan_minuman"},
        {"amount": 30000, "category": "makanan_minuman"},
        {"amount": 28000, "category": "makanan_minuman"},
        {"amount": 26000, "category": "makanan_minuman"},
        {"amount": 250000, "category": "makanan_minuman"},  # outlier
    ]
    results = detect_anomalies(_synthetic, min_samples=4)
    outlier_amounts = [r["amount"] for r in results]
    assert 250000 in outlier_amounts, f"Expected 250000 flagged, got {outlier_amounts}"
    normal_flagged = [a for a in outlier_amounts if a != 250000]
    print(f"[SELF-TEST] outliers={outlier_amounts}  normal_flagged={normal_flagged}")
    for r in results:
        print(f"  amount={r['amount']}  score={r['anomaly_score']:.4f}  reason={r['anomaly_reason']}")

    # Edge cases
    assert detect_anomalies([]) == []
    assert detect_anomalies([{"amount": 0}]) == []
    assert detect_anomalies([{"amount": 100}]) == []   
    print("[SELF-TEST] Edge cases passed.")
