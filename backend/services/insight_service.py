"""Insight service — generates dashboard summaries from saved transactions."""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from ..schemas import (
    AnomalyTransaction,
    CategoryBreakdownItem,
    InsightResponse,
    InsightSummary,
    RecentTransaction,
    Recommendation,
)
from .transaction_service import get_all_transactions

logger = logging.getLogger(__name__)

CATEGORY_DISPLAY: dict[str, str] = {
    "makanan_minuman": "Makanan & Minuman",
    "transportasi": "Transportasi",
    "belanja": "Belanja & Retail",
    "hiburan": "Hiburan & Wisata",
    "kesehatan": "Kesehatan",
    "pendidikan": "Pendidikan",
    "tagihan": "Tagihan & Utilitas",
    "lainnya": "Lainnya",
}

_ATTENTION_LEVELS = {
    "tinggi": "Perlu perhatian segera",
    "sedang": "Perlu diperiksa",
    "rendah": "Untuk ditinjau",
}


def _cat_display(cat: str) -> str:
    return CATEGORY_DISPLAY.get(cat, cat.replace("_", " ").title())


def get_insights(monthly_income: float = 0.0) -> InsightResponse:
    transactions = get_all_transactions()
    txs = [t.model_dump() for t in transactions]

    if not txs:
        return InsightResponse(
            summary=InsightSummary(),
            category_breakdown=[],
            recent_transactions=[],
            recommendations=[],
            transactions_to_review=[],
            budget_comparison={},
        )

    total_expense = sum(t["amount"] for t in txs)
    count = len(txs)
    avg = total_expense / count if count else 0.0

    cat_counter: dict[str, float] = {}
    for t in txs:
        cat_counter[t["category"]] = cat_counter.get(t["category"], 0.0) + t["amount"]

    top_cat = max(cat_counter, key=lambda k: cat_counter[k]) if cat_counter else ""
    merchant_counter: Counter = Counter(t["merchant"] for t in txs if t.get("merchant"))
    top_merchant = merchant_counter.most_common(1)[0][0] if merchant_counter else ""

    summary = InsightSummary(
        total_expense=total_expense,
        transaction_count=count,
        average_transaction=avg,
        top_category=top_cat,
        top_category_display=_cat_display(top_cat),
        top_merchant=top_merchant,
    )

    category_breakdown = []
    for cat, total in sorted(cat_counter.items(), key=lambda x: x[1], reverse=True):
        pct = (total / total_expense * 100) if total_expense else 0.0
        cat_tx_count = sum(1 for t in txs if t["category"] == cat)
        category_breakdown.append(
            CategoryBreakdownItem(
                category=cat,
                category_display=_cat_display(cat),
                total=total,
                count=cat_tx_count,
                percentage=round(pct, 1),
            )
        )

    sorted_txs = sorted(txs, key=lambda t: t.get("saved_at", ""), reverse=True)
    recent_transactions = [
        RecentTransaction(
            id=t["id"],
            merchant=t["merchant"],
            amount=t["amount"],
            date=t["date"],
            category=t["category"],
            category_display=_cat_display(t["category"]),
            source=t["source"],
        )
        for t in sorted_txs[:10]
    ]

    # Try to use recommendation module; fall back gracefully if unavailable
    recommendations: list[Recommendation] = []
    transactions_to_review: list[AnomalyTransaction] = []
    budget_comparison: dict[str, Any] = {}

    try:
        from recommendation import analyze_budget, detect_anomalies, BUCKET_DISPLAY, IDEAL_RATIO

        income = monthly_income if monthly_income > 0 else total_expense * 2
        if income > 0:
            budget_result = analyze_budget(txs, income)
            actual = budget_result.get("actual_ratio", {})
            ideal = budget_result.get("ideal_ratio", {})
            # Build bucket comparison for frontend
            buckets: dict[str, Any] = {}
            for key in ("kebutuhan", "keinginan", "tabungan"):
                buckets[key] = {
                    "actual": budget_result.get("bucket_totals", {}).get(key, 0.0),
                    "actual_ratio": actual.get(key, 0.0),
                    "ideal_ratio": ideal.get(key, 0.0),
                }
            # tabungan amount
            buckets["tabungan"]["actual"] = budget_result.get("tabungan", 0.0)

            budget_comparison = {
                "monthly_income": income,
                "total_spent": budget_result.get("total_spent", 0.0),
                "buckets": buckets,
            }
            # recommendations is a list[str] in budget_rule.py
            for rec_str in budget_result.get("recommendations", []):
                if isinstance(rec_str, str):
                    recommendations.append(
                        Recommendation(bucket="", message=rec_str, detail="")
                    )
                elif isinstance(rec_str, dict):
                    recommendations.append(
                        Recommendation(
                            bucket=rec_str.get("bucket", ""),
                            message=rec_str.get("message", ""),
                            detail=rec_str.get("detail", ""),
                        )
                    )
    except Exception as exc:
        logger.warning("Budget analysis unavailable: %s", exc)

    try:
        from recommendation import detect_anomalies

        anomaly_results = detect_anomalies(txs)
        # detect_anomalies returns augmented transaction dicts (is_anomaly=True)
        # anomaly_score is from IsolationForest: more negative = more anomalous
        for result in anomaly_results:
            raw_score = float(result.get("anomaly_score", 0.0))
            # Convert: -0.5 or below → tinggi, -0.3 to -0.5 → sedang, else → rendah
            if raw_score <= -0.5:
                level = "tinggi"
            elif raw_score <= -0.3:
                level = "sedang"
            else:
                level = "rendah"

            cat = str(result.get("category", "lainnya"))
            transactions_to_review.append(
                AnomalyTransaction(
                    id=int(result.get("id", 0)),
                    merchant=str(result.get("merchant", "")),
                    amount=float(result.get("amount", 0)),
                    date=str(result.get("date", "")),
                    category=cat,
                    category_display=_cat_display(cat),
                    anomaly_reason=result.get("anomaly_reason", "Nilai tidak biasa"),
                    attention_level=level,
                )
            )
    except Exception as exc:
        logger.warning("Anomaly detection unavailable: %s", exc)

    return InsightResponse(
        summary=summary,
        category_breakdown=category_breakdown,
        recent_transactions=recent_transactions,
        recommendations=recommendations,
        transactions_to_review=transactions_to_review,
        budget_comparison=budget_comparison,
    )
