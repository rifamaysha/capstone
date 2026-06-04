"""Batch parser evaluation for Smart Personal Expense.

Scans local project image folders, runs the existing OCR/parser pipeline when
available, and writes qualitative reports to tmp/. This is intentionally a
diagnostic utility, not an app endpoint, not imported by FastAPI runtime, and
not training code.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from importlib.util import find_spec
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.extraction_service import (  # noqa: E402
    CATEGORY_DISPLAY,
    DOC_TYPE_LABEL,
    _category_from_merchant,
    _classify_category,
    _derive_status,
    _detect_route_from_lines,
    _map_document_type,
)
from extraction_postprocessor import (  # noqa: E402
    classify_document_type,
    postprocess,
)
from mbanking_inference import MBankingParser  # noqa: E402


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SEARCH_ROOTS = ("tmp", "dataset", "data", "data_processed")


def collect_images(limit: int) -> list[Path]:
    images: list[Path] = []
    seen: set[Path] = set()
    for name in SEARCH_ROOTS:
        root = PROJECT_ROOT / name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            images.append(path)
            if len(images) >= limit:
                return images
    return images


def suspected_error(row: dict[str, Any]) -> tuple[str, str]:
    warnings = " ".join(str(w) for w in row.get("warnings", []))
    if row.get("ocr_error"):
        return "ocr_unavailable", "Pastikan EasyOCR/model OCR tersedia di environment yang dipakai."
    if not row.get("raw_ocr_preview"):
        return "empty_ocr", "Cek preprocessing gambar atau fallback OCR."
    if row.get("status") == "failed":
        return "no_useful_fields", "Periksa route auto dan fallback parser umum."
    if not row.get("merchant") or str(row.get("merchant")).lower() == "merchant tidak terdeteksi":
        return "merchant_missing", "Tambahkan kandidat merchant dari label/header yang valid."
    if float(row.get("amount") or 0) <= 0:
        return "amount_missing", "Periksa regex nominal dan konteks Total/Nominal."
    if not row.get("date"):
        return "date_missing", "Periksa regex tanggal dan konteks label tanggal."
    if "perlu" in warnings.lower() or row.get("status") == "needs_review":
        return "low_confidence", "Field sudah terbaca sebagian; tingkatkan confidence/context scoring."
    return "ok", "Tidak ada error utama yang terlihat dari heuristic report."


def evaluate_image(parser: MBankingParser | None, image_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    rel = image_path.relative_to(PROJECT_ROOT)
    row: dict[str, Any] = {
        "filename": image_path.name,
        "path": str(rel),
        "predicted_document_type": "unknown",
        "document_type_label": DOC_TYPE_LABEL["unknown"],
        "document_type_internal": "",
        "route_used": "unknown",
        "route_confidence": 0.0,
        "merchant": "",
        "amount": 0.0,
        "date": "",
        "category": "lainnya",
        "category_display": CATEGORY_DISPLAY["lainnya"],
        "status": "failed",
        "success": False,
        "warnings": [],
        "raw_ocr_preview": "",
        "debug_trace_preview": "",
        "processing_time": 0.0,
        "ocr_error": "",
    }

    try:
        if parser is None:
            raise RuntimeError("EasyOCR is not available in this Python environment")

        lines = parser.extract_text_lines(image_path)
        row["raw_ocr_preview"] = "\n".join(lines)[:900]

        route, route_conf = _detect_route_from_lines(lines)
        internal = classify_document_type(lines)
        public, label, mapped_internal = _map_document_type(internal, route)

        if route == "screenshot":
            raw = parser.parse(image_path, return_raw=True, pre_ocr_lines=lines)
            post = postprocess(
                raw.get("ocr_lines", lines),
                route_type="screenshot",
                existing_merchant=raw.get("recipient") or "",
                existing_amount=float(raw.get("amount") or 0),
                existing_date=raw.get("date") or "",
            )
        else:
            post = postprocess(lines, route_type="receipt")

        merchant = post.get("merchant", "") or ""
        amount = float(post.get("amount") or 0)
        date = post.get("date", "") or ""
        field_conf = post.get("field_confidence", {}) or {}
        category_result = _category_from_merchant(merchant)
        if not category_result:
            category_result = _classify_category(merchant or row["raw_ocr_preview"])
            if not merchant and row["raw_ocr_preview"]:
                category_result["confidence"] = min(category_result["confidence"], 0.42)
        category = category_result.get("label", "lainnya")
        success, status = _derive_status(
            merchant=merchant,
            amount=amount,
            date=date,
            doc_type_conf=route_conf,
            field_conf={
                "merchant": float(field_conf.get("merchant", 0) or 0),
                "amount": float(field_conf.get("amount", 0) or 0),
                "date": float(field_conf.get("date", 0) or 0),
            },
        )

        row.update({
            "predicted_document_type": public,
            "document_type_label": label,
            "document_type_internal": mapped_internal,
            "route_used": route,
            "route_confidence": round(float(route_conf), 3),
            "merchant": merchant,
            "amount": amount,
            "date": date,
            "category": category,
            "category_display": CATEGORY_DISPLAY.get(category, "Lainnya"),
            "status": status,
            "success": success,
            "warnings": post.get("warnings", []) or [],
            "debug_trace_preview": (post.get("debug_trace", "") or "")[:1200],
        })
    except Exception as exc:  # noqa: BLE001
        row["ocr_error"] = f"{type(exc).__name__}: {exc}"
        row["warnings"] = [row["ocr_error"]]
    finally:
        row["processing_time"] = round(time.perf_counter() - started, 2)

    err, fix = suspected_error(row)
    row["suspected_error_type"] = err
    row["suggested_fix"] = fix
    return row


def write_reports(rows: list[dict[str, Any]], easyocr_available: bool, parser_error: str) -> None:
    out_dir = PROJECT_ROOT / "tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "parser_eval_result.json"
    summary_path = out_dir / "parser_eval_summary.md"
    errors_path = out_dir / "parser_eval_errors.csv"

    status_counts = Counter(row["status"] for row in rows)
    type_counts = Counter(row["predicted_document_type"] for row in rows)
    error_counts = Counter(row["suspected_error_type"] for row in rows)
    by_folder: dict[str, int] = defaultdict(int)
    for row in rows:
        by_folder[str(Path(row["path"]).parent)] += 1

    completeness = {
        "merchant": sum(1 for r in rows if r.get("merchant") and str(r.get("merchant")).lower() != "merchant tidak terdeteksi"),
        "amount": sum(1 for r in rows if float(r.get("amount") or 0) > 0),
        "date": sum(1 for r in rows if r.get("date")),
        "category": sum(1 for r in rows if r.get("category") and r.get("category") != "lainnya"),
    }
    avg_time = round(sum(float(r.get("processing_time") or 0) for r in rows) / max(len(rows), 1), 2)
    times = [float(r.get("processing_time") or 0) for r in rows]
    median_time = round(statistics.median(times), 2) if times else 0.0
    max_time = round(max(times), 2) if times else 0.0
    slowest = sorted(rows, key=lambda r: float(r.get("processing_time") or 0), reverse=True)[:10]
    missing_counts = {
        "merchant_missing": len(rows) - completeness["merchant"],
        "amount_missing": len(rows) - completeness["amount"],
        "date_missing": len(rows) - completeness["date"],
    }

    payload = {
        "summary": {
            "total_samples": len(rows),
            "easyocr_available": easyocr_available,
            "parser_error": parser_error,
            "status_distribution": dict(status_counts),
            "document_type_distribution": dict(type_counts),
            "field_completeness": completeness,
            "missing_counts": missing_counts,
            "suspected_error_distribution": dict(error_counts),
            "average_processing_time_seconds": avg_time,
            "median_processing_time_seconds": median_time,
            "max_processing_time_seconds": max_time,
            "sample_folders": dict(sorted(by_folder.items())),
            "slowest_samples": [
                {
                    "path": row.get("path"),
                    "processing_time": row.get("processing_time"),
                    "status": row.get("status"),
                    "suspected_error_type": row.get("suspected_error_type"),
                }
                for row in slowest
            ],
        },
        "results": rows,
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with errors_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path", "predicted_document_type", "document_type_label", "route_used",
                "merchant", "amount", "date", "category", "status", "warnings",
                "elapsed_seconds", "debug_trace", "raw_ocr_preview",
                "suspected_error_type", "suggested_fix",
            ],
        )
        writer.writeheader()
        for row in rows:
            if row["suspected_error_type"] == "ok":
                continue
            writer.writerow({
                "path": row["path"],
                "predicted_document_type": row["predicted_document_type"],
                "document_type_label": row["document_type_label"],
                "route_used": row["route_used"],
                "merchant": row["merchant"],
                "amount": row["amount"],
                "date": row["date"],
                "category": row["category"],
                "status": row["status"],
                "warnings": " | ".join(str(w) for w in row.get("warnings", [])),
                "elapsed_seconds": row.get("processing_time", 0),
                "debug_trace": row.get("debug_trace_preview", "").replace("\n", " / "),
                "raw_ocr_preview": row.get("raw_ocr_preview", "").replace("\n", " / "),
                "suspected_error_type": row["suspected_error_type"],
                "suggested_fix": row["suggested_fix"],
            })

    lines = [
        "# Parser Evaluation Summary",
        "",
        f"- Total sample: {len(rows)}",
        f"- EasyOCR tersedia: {easyocr_available}",
        f"- Catatan parser: {parser_error or '-'}",
        f"- Rata-rata waktu proses: {avg_time}s",
        f"- Median waktu proses: {median_time}s",
        f"- Maks waktu proses: {max_time}s",
        "",
        "## Status",
        *[f"- {key}: {value}" for key, value in sorted(status_counts.items())],
        "",
        "## Document Type",
        *[f"- {key}: {value}" for key, value in sorted(type_counts.items())],
        "",
        "## Field Completeness",
        *[f"- {key}: {value}/{len(rows)}" for key, value in completeness.items()],
        "",
        "## Missing Field Count",
        *[f"- {key}: {value}" for key, value in missing_counts.items()],
        "",
        "## Error Pattern",
        *[f"- {key}: {value}" for key, value in sorted(error_counts.items())],
        "",
        "## 10 Slowest Samples",
        *[
            f"- {row.get('processing_time')}s `{row.get('path')}` ({row.get('status')}, {row.get('suspected_error_type')})"
            for row in slowest
        ],
        "",
        "## Fine-tuning Decision",
        "Belum perlu retraining otomatis. Jika raw OCR sudah berisi teks benar tetapi field salah, prioritasnya tetap parser/postprocessor. Jika banyak raw OCR kosong atau rusak, baru pertimbangkan preprocessing/OCR fallback atau fine-tuning DONUT secara terpisah.",
        "",
        f"Detail JSON: `{result_path.as_posix()}`",
        f"Error CSV: `{errors_path.as_posix()}`",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {result_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {errors_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local parser batch evaluation.")
    parser.add_argument("--limit", type=int, default=80, help="Maximum local images to evaluate.")
    parser.add_argument("--max-seconds", type=int, default=240, help="Stop after this many seconds and write partial reports.")
    args = parser.parse_args()

    images = collect_images(max(1, args.limit))
    easyocr_available = find_spec("easyocr") is not None
    parser_error = ""
    ocr_parser: MBankingParser | None = None
    if easyocr_available:
        try:
            ocr_parser = MBankingParser()
        except Exception as exc:  # noqa: BLE001
            parser_error = f"{type(exc).__name__}: {exc}"
            ocr_parser = None
    else:
        parser_error = "easyocr is not installed in this Python environment"

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for image_path in images:
        if rows and time.perf_counter() - started > args.max_seconds:
            rows.append({
                "filename": image_path.name,
                "path": str(image_path.relative_to(PROJECT_ROOT)),
                "predicted_document_type": "unknown",
                "document_type_label": DOC_TYPE_LABEL["unknown"],
                "document_type_internal": "",
                "route_used": "not_run",
                "route_confidence": 0.0,
                "merchant": "",
                "amount": 0.0,
                "date": "",
                "category": "lainnya",
                "category_display": CATEGORY_DISPLAY["lainnya"],
                "status": "failed",
                "success": False,
                "warnings": [f"Runtime guard stopped after {args.max_seconds}s."],
                "raw_ocr_preview": "",
                "debug_trace_preview": "",
                "processing_time": 0.0,
                "ocr_error": "runtime_guard",
                "suspected_error_type": "runtime_guard",
                "suggested_fix": "Kurangi limit atau jalankan evaluasi saat resource CPU lebih longgar.",
            })
            break
        rows.append(evaluate_image(ocr_parser, image_path))
    write_reports(rows, easyocr_available, parser_error)


if __name__ == "__main__":
    main()
