"""Pydantic models for Smart Personal Expense API."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class ConfidenceFields(BaseModel):
    merchant: float = 0.0
    amount: float = 0.0
    date: float = 0.0
    category: float = 0.0


class ExtractionResponse(BaseModel):
    # Core extraction result
    merchant: str = ""
    amount: float = 0.0
    date: str = ""
    category: str = "lainnya"
    category_display: str = "Lainnya"

    # Document classification
    document_type: str = "unknown"
    document_type_label: str = "Perlu dicek manual"
    document_type_internal: str = ""
    document_type_confidence: float = 0.0
    document_type_source: str = "manual"   # "manual" | "auto"

    # Quality signals
    confidence: ConfidenceFields = Field(default_factory=ConfidenceFields)
    warnings: list[str] = Field(default_factory=list)

    # Status
    success: bool = True
    status: str = "extracted"   # "extracted" | "needs_review" | "failed"

    # Debug — sent to client but hidden from UI by default
    debug_trace: str = ""


class TransactionUpdate(BaseModel):
    category: str

    @field_validator("category")
    @classmethod
    def category_valid(cls, v: str) -> str:
        valid = {
            "makanan_minuman", "transportasi", "belanja", "hiburan",
            "kesehatan", "pendidikan", "tagihan", "lainnya",
        }
        if v not in valid:
            raise ValueError(f"kategori tidak valid: {v}")
        return v


class TransactionCreate(BaseModel):
    merchant: str
    amount: float
    date: str = ""
    category: str = "lainnya"
    source: str = "receipt"
    notes: str = ""

    @field_validator("amount")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("amount tidak boleh negatif")
        return v

    @field_validator("merchant")
    @classmethod
    def merchant_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("merchant tidak boleh kosong")
        return v


class TransactionOut(BaseModel):
    id: int
    merchant: str
    amount: float
    date: str
    category: str
    category_display: str = ""
    source: str
    notes: str = ""
    saved_at: str


class TransactionListResponse(BaseModel):
    transactions: list[TransactionOut]


class CategoryBreakdownItem(BaseModel):
    category: str
    category_display: str
    total: float
    count: int
    percentage: float = 0.0


class RecentTransaction(BaseModel):
    id: int
    merchant: str
    amount: float
    date: str
    category: str
    category_display: str
    source: str


class AnomalyTransaction(BaseModel):
    id: int
    merchant: str
    amount: float
    date: str
    category: str
    category_display: str
    anomaly_reason: str = ""
    attention_level: str = "rendah"


class Recommendation(BaseModel):
    bucket: str
    message: str
    detail: str = ""


class InsightSummary(BaseModel):
    total_expense: float = 0.0
    transaction_count: int = 0
    average_transaction: float = 0.0
    top_category: str = ""
    top_category_display: str = ""
    top_merchant: str = ""
    monthly_income: float = 0.0
    remaining_balance: float = 0.0


class DailyExpenseItem(BaseModel):
    date: str
    total: float
    count: int


class InsightResponse(BaseModel):
    summary: InsightSummary
    category_breakdown: list[CategoryBreakdownItem] = Field(default_factory=list)
    recent_transactions: list[RecentTransaction] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    transactions_to_review: list[AnomalyTransaction] = Field(default_factory=list)
    budget_comparison: dict[str, Any] = Field(default_factory=dict)
    daily_expenses: list[DailyExpenseItem] = Field(default_factory=list)
