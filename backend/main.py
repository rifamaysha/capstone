"""Smart Personal Expense — FastAPI backend.

Run with:
    uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    ExtractionResponse,
    InsightResponse,
    TransactionCreate,
    TransactionListResponse,
    TransactionOut,
    TransactionUpdate,
)
from .services import extraction_service, insight_service, transaction_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("smart_expense.api")

for _lib in ("transformers", "huggingface_hub", "httpx", "easyocr"):
    logging.getLogger(_lib).setLevel(logging.ERROR)

app = FastAPI(
    title="Smart Personal Expense API",
    description="Backend for Smart Personal Expense — React + FastAPI architecture.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TMP_DIR = Path(__file__).parent.parent / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

_VALID_TYPES = {"receipt", "screenshot", "auto"}


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "app": "Smart Personal Expense",
        "backend": "FastAPI",
    }


# ── Extraction ────────────────────────────────────────────────────────────────

@app.post("/extract", response_model=ExtractionResponse)
async def extract_transaction(
    file: UploadFile = File(...),
    selected_type: str = Form("auto"),
) -> ExtractionResponse:
    """Upload a receipt/screenshot image; OCR and return extracted fields.

    selected_type: "receipt" | "screenshot" | "auto"  (default: "auto")
    """
    stype = (selected_type or "auto").strip().lower()
    if stype not in _VALID_TYPES:
        stype = "auto"

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    tmp_file = TMP_DIR / f"upload_{uuid.uuid4().hex[:8]}{suffix}"

    try:
        with open(tmp_file, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
    except Exception as exc:
        logger.error("Failed to save upload: %s", exc)
        raise HTTPException(status_code=500, detail="Gagal menyimpan file yang diunggah.")

    try:
        result = extraction_service.extract_from_image(tmp_file, stype)
    except Exception as exc:
        # Outer safety net: extraction_service should never raise, but just in case
        logger.exception("Extraction service unhandled error: %s", exc)
        result = {
            "merchant": "",
            "amount": 0.0,
            "date": "",
            "category": "lainnya",
            "category_display": "Lainnya",
            "document_type": "unknown",
            "document_type_label": "Perlu dicek manual",
            "document_type_internal": "",
            "document_type_confidence": 0.0,
            "document_type_source": "auto",
            "confidence": {"merchant": 0.0, "amount": 0.0, "date": 0.0, "category": 0.0},
            "warnings": ["Layanan pembacaan mengalami gangguan. Isi data secara manual."],
            "success": False,
            "status": "failed",
            "debug_trace": "",
        }
    finally:
        try:
            tmp_file.unlink(missing_ok=True)
        except Exception:
            pass

    return ExtractionResponse(**result)


# ── Transactions ──────────────────────────────────────────────────────────────

@app.get("/transactions", response_model=TransactionListResponse)
def get_transactions() -> TransactionListResponse:
    return TransactionListResponse(transactions=transaction_service.get_all_transactions())


@app.post("/transactions", response_model=TransactionOut, status_code=201)
def create_transaction(payload: TransactionCreate) -> TransactionOut:
    try:
        return transaction_service.save_transaction(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to save transaction: %s", exc)
        raise HTTPException(status_code=500, detail="Gagal menyimpan transaksi.")


@app.patch("/transactions/{transaction_id}", response_model=TransactionOut)
def update_transaction(transaction_id: int, payload: TransactionUpdate) -> TransactionOut:
    try:
        return transaction_service.update_transaction_category(transaction_id, payload.category)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.delete("/transactions/{transaction_id}")
def delete_transaction(transaction_id: int) -> dict:
    try:
        return transaction_service.delete_transaction_by_id(transaction_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.delete("/transactions")
def delete_transactions() -> dict:
    return transaction_service.delete_all_transactions()


# ── Insights ──────────────────────────────────────────────────────────────────

@app.get("/insights", response_model=InsightResponse)
def get_insights(monthly_income: float = 0.0) -> InsightResponse:
    try:
        return insight_service.get_insights(monthly_income=monthly_income)
    except Exception as exc:
        logger.exception("Insight service error: %s", exc)
        raise HTTPException(status_code=500, detail="Gagal memuat insight.")
