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

# Minimum total transactions required before anomaly detection runs.
# Below this, the median/mean baseline is too unstable — IsolationForest is
# forced to flag at least one item (because contamination clamps to 1/n),
# producing false alarms on completely normal data.
MIN_TRANSACTIONS_FOR_DETECTION = 10


def _format_rp(n: float) -> str:
    """Format number as 'Rp 1.234.567' (Indonesian thousand separator)."""
    return f"Rp {int(round(n)):,}".replace(",", ".")


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

        # Only flag transactions that are unusually HIGH.
        # IsolationForest also flags transactions that are unusually LOW
        # (e.g. a Rp 25rb coffee when the median is Rp 19jt) — but those are
        # almost never the user's "suspicious" intent. We skip them.
        if baseline_median > 0 and amt <= baseline_median:
            continue

        tx_out = dict(tx)
        tx_out["anomaly_score"]    = float(score)
        tx_out["is_anomaly"]       = True
        tx_out["baseline_median"]  = round(baseline_median, 2)
        tx_out["baseline_mean"]    = round(baseline_mean, 2)
        tx_out["baseline_scope"]   = scope

        # User-friendly Indonesian reason (only HIGH-value variants now)
        if baseline_median > 0 and amt > baseline_median * 3:
            reason = (
                f"Nominal jauh lebih tinggi dari pola {cat_label} "
                f"(median {_format_rp(baseline_median)})"
            )
        else:
            reason = (
                f"Nominal di atas pola {cat_label} "
                f"(median {_format_rp(baseline_median)})"
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

    # ---- 1a. Minimum sample size guard ----
    # With too few transactions, the algorithm produces unreliable results:
    #   - IsolationForest is forced to flag ~1/n transactions due to
    #     contamination clamping, creating false alarms on normal data
    #   - The median/mean baseline shifts dramatically per transaction
    # Better to show nothing than show misleading "anomalies".
    if len(valid) < MIN_TRANSACTIONS_FOR_DETECTION:
        logger.info(
            "Skipping anomaly detection: %d valid transactions (need >= %d)",
            len(valid), MIN_TRANSACTIONS_FOR_DETECTION,
        )
        return []

    # ---- 1b. Rule-based: catch obvious "way too big" transactions ----
    # Flag any transaction whose amount is > 3x the overall mean. Always
    # runs alongside IsolationForest as a safety net for clear outliers.
    amounts = [float(t.get("amount") or 0) for t in valid]
    mean_amt = sum(amounts) / len(amounts)
    rule_anomalies: list[dict[str, Any]] = []

    if mean_amt > 0:
        sorted_amounts = sorted(amounts)
        median_amt = sorted_amounts[len(amounts) // 2]
        for t in valid:
            amt = float(t.get("amount") or 0)
            if amt > mean_amt * 3:
                tx_out = dict(t)
                tx_out["anomaly_score"] = -0.6
                tx_out["is_anomaly"] = True
                tx_out["baseline_median"] = median_amt
                tx_out["baseline_mean"] = round(mean_amt, 2)
                tx_out["baseline_scope"] = "rule_based"
                tx_out["anomaly_reason"] = (
                    f"Nominal {_format_rp(amt)} jauh lebih tinggi dari rata-rata "
                    f"transaksi lain ({_format_rp(mean_amt)})"
                )
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
    # Test 1: with >= 10 transactions, the high outlier should be flagged.
    _synthetic = [
        {"amount": 25000, "category": "makanan_minuman"},
        {"amount": 27000, "category": "makanan_minuman"},
        {"amount": 30000, "category": "makanan_minuman"},
        {"amount": 28000, "category": "makanan_minuman"},
        {"amount": 26000, "category": "makanan_minuman"},
        {"amount": 24000, "category": "makanan_minuman"},
        {"amount": 29000, "category": "makanan_minuman"},
        {"amount": 31000, "category": "makanan_minuman"},
        {"amount": 23000, "category": "makanan_minuman"},
        {"amount": 250000, "category": "makanan_minuman"},  # high outlier
    ]
    results = detect_anomalies(_synthetic, min_samples=4)
    outlier_amounts = [r["amount"] for r in results]
    assert 250000 in outlier_amounts, f"Expected 250000 flagged, got {outlier_amounts}"
    # Low values must NEVER be flagged
    low_flagged = [a for a in outlier_amounts if a < 28000]
    assert not low_flagged, f"Low values should not be flagged: {low_flagged}"
    print(f"[SELF-TEST] outliers={outlier_amounts}")
    for r in results:
        print(f"  amount={r['amount']}  score={r['anomaly_score']:.4f}  reason={r['anomaly_reason']}")

    # Test 2: small dataset (< 10) returns empty — no false alarms
    _small = [
        {"amount": 2_400_000, "category": "makanan_minuman", "id": 1},
        {"amount": 30_000_000, "category": "belanja", "id": 2},
        {"amount": 19_000_000, "category": "makanan_minuman", "id": 3},
    ]
    small_results = detect_anomalies(_small)
    assert small_results == [], f"Expected no anomalies for <10 txs, got {small_results}"
    print(f"[SELF-TEST] small dataset (n=3) → no false alarms ✓")

    # Edge cases
    assert detect_anomalies([]) == []
    assert detect_anomalies([{"amount": 0}]) == []
    assert detect_anomalies([{"amount": 100}]) == []
    print("[SELF-TEST] Edge cases passed.")
