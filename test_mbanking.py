"""Test parser M-Banking di sample dari test set."""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from importlib.util import find_spec
from pathlib import Path

from mbanking_inference import MBankingParser, _extract_amounts
from extraction_postprocessor import (
    classify_document_type,
    detect_receipt_currency,
    extract_amount_mbanking,
    extract_merchant_mbanking,
    extract_merchant_receipt,
    postprocess,
)
from backend.services.extraction_service import (
    _classify_category,
    _derive_status,
    _detect_route_from_lines,
)


SMOKE_CATEGORY_DISPLAY = {
    "makanan_minuman": "Makanan & Minuman",
    "transportasi": "Transportasi",
    "belanja": "Belanja & Retail",
    "hiburan": "Hiburan & Wisata",
    "kesehatan": "Kesehatan",
    "pendidikan": "Pendidikan",
    "tagihan": "Tagihan & Utilitas",
    "lainnya": "Lainnya",
}


def _smoke_category(merchant: str, raw_text: str, merchant_conf: float = 0.0) -> tuple[str, float, str]:
    text = f"{merchant or ''}\n{raw_text or ''}".lower()
    merchant_lower = (merchant or "").strip().lower()
    unresolved = not merchant_lower or merchant_lower == "merchant tidak terdeteksi"
    groups = [
        ("tagihan", ("pln", "listrik", "pdam", "pulsa", "paket data", "internet", "wifi", "telkom", "bill", "utility")),
        ("transportasi", ("spbu", "shell", "parkir", "parking", "toll", "tol", "gojek", "grab", "ride", "transport")),
        ("belanja", ("indomaret", "idm", "alfamart", "alfamidi", "minimarket", "supermarket", "retail", "toko", "copy", "digital")),
        ("makanan_minuman", ("kantin", "warteg", "warung makan", "martabak", "batagor", "cimol", "kopi", "coffee", "cafe", "restaurant", "restoran", "ayam", "bakso", "mie", "nasi", "dimsum", "tahu", "crispy", "drink", "food", "mixue", "harvest cakes", "bread", "bakery", "gacoan")),
        ("lainnya", ("laundry", "dry clean", "dryclean", "laundromat", "cuci kiloan")),
    ]
    for label, terms in groups:
        for term in terms:
            if term in text:
                conf = 0.48 if unresolved else (0.82 if merchant_conf >= 0.70 else 0.62)
                return label, conf, f"smoke_keyword:{term}"
    return "lainnya", 0.25 if unresolved else 0.40, "smoke_fallback"


def _image_group_from_path(path: Path) -> str:
    lower = str(path).lower()
    name = path.name.lower()
    if any(token in lower for token in ("qris", "qr", "payment", "pembayaran")):
        return "compact_qr_screenshots"
    if any(token in lower for token in ("mbanking", "m-banking", "banking", "ewallet", "gopay", "shopeepay", "ovo", "dana")):
        return "mbanking_detail_screenshots"
    if any(token in lower for token in ("dataset_hf", "kaggle", "receipt", "struk", "cord", "nota")):
        return "receipt_photos"
    if name.startswith(("img_", "000")):
        return "receipt_photos"
    return "unknown_difficult"


def _selected_route_from_doc_type(doc_type: str, path_group: str) -> str:
    if doc_type in {"compact_qr_card", "mbanking_transaction_detail", "ewallet_receipt_screen"}:
        return "screenshot"
    if doc_type == "photo_receipt":
        return "receipt"
    if path_group in {"compact_qr_screenshots", "mbanking_detail_screenshots"}:
        return "screenshot"
    if path_group == "receipt_photos":
        return "receipt"
    return "uncertain"


def _collect_smoke_images(project_root: Path, limit_per_group: int) -> dict[str, list[Path]]:
    roots = [
        project_root / "tmp",
        project_root / "dataset",
        project_root / "data",
        project_root / "data_processed",
    ]
    grouped: dict[str, list[Path]] = defaultdict(list)
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            group = _image_group_from_path(path)
            if len(grouped[group]) < limit_per_group:
                grouped[group].append(path)
    return dict(grouped)


def run_dataset_smoke(limit_per_group: int = 20, max_runtime_seconds: int = 90) -> None:
    project_root = Path(__file__).resolve().parent
    out_path = project_root / "tmp" / "parser_smoke_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    grouped = _collect_smoke_images(project_root, limit_per_group)
    results: list[dict] = []
    if not grouped:
        payload = {"summary": {"total": 0, "note": "No images found in tmp/dataset/data/data_processed."}, "results": []}
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"No image folders found for smoke test. Wrote {out_path}")
        return

    easyocr_available = find_spec("easyocr") is not None
    parser: MBankingParser | None = None
    if easyocr_available:
        try:
            parser = MBankingParser()
        except Exception as exc:  # noqa: BLE001
            easyocr_available = False
            parser_error = f"{type(exc).__name__}: {exc}"
        else:
            parser_error = ""
    else:
        parser_error = "easyocr is not installed in this Python environment"

    started = time.time()
    for group, paths in grouped.items():
        for image_path in paths:
            elapsed_total = time.time() - started
            if elapsed_total > max_runtime_seconds:
                results.append({
                    "filename": image_path.name,
                    "folder": str(image_path.parent.relative_to(project_root)),
                    "path_group": group,
                    "detected_doc_type": "not_run",
                    "selected_route": "not_run",
                    "merchant": "",
                    "amount": 0.0,
                    "date": "",
                    "category": "lainnya",
                    "category_display": SMOKE_CATEGORY_DISPLAY["lainnya"],
                    "confidence": {"merchant": 0.0, "amount": 0.0, "date": 0.0, "category": 0.0},
                    "warnings": [f"Runtime guard stopped after {max_runtime_seconds}s."],
                    "processing_time": 0.0,
                    "debug_trace_summary": "runtime_guard",
                    "raw_ocr_preview": "",
                })
                break

            t0 = time.time()
            raw_lines: list[str] = []
            warnings: list[str] = []
            detected_doc_type = "other_unknown"
            selected_route = _selected_route_from_doc_type(detected_doc_type, group)
            merchant = ""
            amount = 0.0
            date = ""
            field_conf = {"merchant": 0.0, "amount": 0.0, "date": 0.0, "category": 0.0}
            trace_summary = ""

            if parser is None:
                warnings.append(f"OCR skipped: {parser_error}")
            else:
                try:
                    raw_lines = parser.extract_text_lines(image_path)
                    detected_doc_type = classify_document_type(raw_lines)
                    selected_route = _selected_route_from_doc_type(detected_doc_type, group)
                    if selected_route == "screenshot":
                        parsed = parser.parse(image_path, return_raw=True, pre_ocr_lines=raw_lines)
                        pp = postprocess(
                            parsed.get("ocr_lines", raw_lines),
                            route_type="screenshot",
                            existing_merchant=parsed.get("recipient") or "",
                            existing_amount=float(parsed.get("amount") or 0),
                            existing_date=parsed.get("date") or "",
                        )
                    else:
                        pp = postprocess(raw_lines, route_type="receipt")
                    merchant = pp["merchant"]
                    amount = float(pp["amount"] or 0)
                    date = pp["date"]
                    field_conf.update(pp["field_confidence"])
                    warnings.extend(pp.get("warnings", []))
                    trace_summary = "\n".join((pp.get("debug_trace") or "").splitlines()[:18])
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"parse_error: {type(exc).__name__}: {exc}")

            category, cat_conf, cat_source = _smoke_category(
                merchant, "\n".join(raw_lines), field_conf.get("merchant", 0.0)
            )
            field_conf["category"] = cat_conf
            processing_time = round(time.time() - t0, 2)
            results.append({
                "filename": image_path.name,
                "folder": str(image_path.parent.relative_to(project_root)),
                "path_group": group,
                "detected_doc_type": detected_doc_type,
                "selected_route": selected_route,
                "merchant": merchant or "Merchant tidak terdeteksi",
                "amount": amount,
                "date": date,
                "category": category,
                "category_display": SMOKE_CATEGORY_DISPLAY.get(category, category),
                "category_source": cat_source,
                "confidence": field_conf,
                "warnings": warnings,
                "processing_time": processing_time,
                "debug_trace_summary": trace_summary or ("ocr_skipped" if parser is None else ""),
                "raw_ocr_preview": "\n".join(raw_lines)[:700],
            })

    by_group = defaultdict(int)
    parsed_count = 0
    for row in results:
        by_group[row["path_group"]] += 1
        if not any(str(w).startswith("OCR skipped") for w in row["warnings"]):
            parsed_count += 1
    payload = {
        "summary": {
            "total": len(results),
            "parsed": parsed_count,
            "groups": dict(by_group),
            "easyocr_available": easyocr_available,
            "parser_error": parser_error,
        },
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDataset smoke summary")
    print("-" * 72)
    print(f"Images recorded: {len(results)} | Parsed with OCR: {parsed_count} | Output: {out_path}")
    for group, count in sorted(by_group.items()):
        print(f"  {group:<30} {count:>3}")
    if parser_error:
        print(f"  OCR note: {parser_error}")


def run_unit_checks() -> None:
    compact_context = "QR Bayar\nPembayaran QR"
    amount_cases = {
        "IDR 10.00000": 10000,
        "IDR 26.00009": 26000,
        "IDR 33.50000": 33500,
        "IDR 94.80009": 94800,
        "IDR 12.00000": 12000,
        "Rp 15.000": 15000,
        "IDR 10.000": 10000,
        "-Rp320.000": 320000,
        "Rp78.750": 78750,
        "Rp 117.657": 117657,
        "IDR 64.000": 64000,
        "IDR 120.960": 120960,
        "IDR 820.000": 820000,
        "IDR 260.009": 26000,
        "IDR 948.009": 94800,
    }
    for raw, expected in amount_cases.items():
        got = _extract_amounts(raw, context=compact_context, doc_type="compact_qr_card")
        assert got and got[0] == expected, f"{raw!r}: got={got}, expected={expected}"

    pp_amount, pp_conf, _ = extract_amount_mbanking([
        "QR Bayar",
        "Pembayaran QR",
        "- IDR 260.009",
        "ke WARUNG TEST QR",
        "936000123456789",
    ])
    assert pp_amount == 26000, f"compact QR postprocessor amount={pp_amount}"
    assert 0.45 <= pp_conf <= 0.90, f"unexpected amount confidence={pp_conf}"

    merchant, merch_conf, _ = extract_merchant_mbanking([
        "QR Bayar",
        "Pembayaran QR",
        "- IDR 26.00009",
        "ke WARUNG TEST QR",
        "936000123456789",
    ])
    assert merchant == "WARUNG TEST QR", merchant
    assert merch_conf >= 0.70, merch_conf

    compact_lines = [
        "QR Bayar", "Pembayaran QR", "- IDR 120.960", "ke QRIS IDM TGT9 SUKABIRUS",
        "936000123456789", "Customer PAN", "9360 **** **** 1234", "Terminal ID",
        "A1234567", "Acquirer Name", "Bank Test", "Rincian Pembayaran",
        "Bagikan", "Selesai", "OK", "Info tambahan",
    ]
    assert classify_document_type(compact_lines) == "compact_qr_card"
    pp = postprocess(compact_lines, route_type="screenshot", existing_merchant="QRIS IDM TGT9 SUKABIRUS")
    assert pp["merchant"] == "QRIS IDM TGT9 SUKABIRUS", pp["debug_trace"]
    assert pp["amount"] == 120960, pp["debug_trace"]
    assert pp["date"] == "", pp["date"]

    alfamart_lines = [
        "Atfamardt",
        "Jl. Contoh No. 1",
        "18.7.2024 08:49",
        "No Struk 123456",
        "Susu UHT 10.000",
        "Roti 8.000",
        "TOTAL",
        "Rp38.000",
        "TUNAI 40.000",
        "KEMBALI 2.000",
    ]
    route, doc_conf = _detect_route_from_lines(alfamart_lines)
    assert route == "receipt", (route, doc_conf)
    receipt_pp = postprocess(alfamart_lines, route_type="receipt")
    assert "Alfamart" in receipt_pp["merchant"], receipt_pp["debug_trace"]
    assert receipt_pp["amount"] == 38000, receipt_pp["debug_trace"]
    assert receipt_pp["date"] == "18/07/2024", receipt_pp["debug_trace"]
    success, status = _derive_status(
        merchant=receipt_pp["merchant"],
        amount=receipt_pp["amount"],
        date=receipt_pp["date"],
        doc_type_conf=doc_conf,
        field_conf=receipt_pp["field_confidence"],
    )
    assert success is True
    assert status in {"extracted", "needs_review"}, status
    category_result = _classify_category("\n".join([receipt_pp["merchant"], *alfamart_lines]))
    assert category_result["label"] == "belanja", category_result

    payment_lines = [
        "Transaction Details",
        "Pembayaran Berhasil",
        "Bayar ke",
        "TOKO SEJAHTERA",
        "Nominal",
        "-Rp105.000",
        "Reference Number",
        "ABC123456789",
        "Date and time",
        "13 May 2026, 11:18",
    ]
    route, _ = _detect_route_from_lines(payment_lines)
    assert route == "screenshot", route
    payment_pp = postprocess(payment_lines, route_type="screenshot")
    assert payment_pp["amount"] == 105000, payment_pp["debug_trace"]
    assert payment_pp["merchant"] == "TOKO SEJAHTERA", payment_pp["debug_trace"]
    assert "reference" not in payment_pp["merchant"].lower()
    assert "transaction" not in payment_pp["merchant"].lower()

    qris_pan_lines = [
        "Rincian Pembayaran",
        "QRIS",
        "Merchant PAN",
        "936000123456789012",
        "Customer PAN",
        "936000987654321012",
        "Terminal ID",
        "A1234567",
        "Merchant Name",
        "WARUNG QR TEST",
        "Total",
        "Rp20.000",
    ]
    route, _ = _detect_route_from_lines(qris_pan_lines)
    assert route == "screenshot", route
    qris_pp = postprocess(qris_pan_lines, route_type="screenshot")
    assert qris_pp["amount"] == 20000, qris_pp["debug_trace"]
    assert qris_pp["merchant"] == "WARUNG QR TEST", qris_pp["debug_trace"]
    assert "pan" not in qris_pp["merchant"].lower()
    assert not any(token in qris_pp["merchant"] for token in ("936000", "A1234567"))

    kept = postprocess(
        ["From", "Transaction ID", "260104-RWBR-DBAXWB", "Date and time", "IDR 64.000"],
        route_type="screenshot",
        existing_merchant="TOKO VALID JAYA",
        existing_amount=64000,
    )
    assert kept["merchant"] == "TOKO VALID JAYA", kept["debug_trace"]

    trimmed, _, _ = extract_merchant_mbanking([
        "Rincian Transaksi",
        "Merchant Name",
        "KEDAI KOPI NYAMAN, Jl. Melati No. 8 Bandung",
        "IDR 64.000",
    ])
    assert trimmed == "KEDAI KOPI NYAMAN", trimmed

    merchant_trim_cases = {
        "The Harvest Cakes, Daan Mogot Jl. Tampak Siring Blok KJ No. 78 Jakarta": "The Harvest Cakes, Daan Mogot",
        "Mixue, Patriot Jakasampurna; Kec. Bekasi Barat": "Mixue, Patriot Jakasampurna",
        "Martabak Legit Group, Tebet Jl. Albarkah 1 No. 8, Tebet, Jakarta": "Martabak Legit Group, Tebet",
        "Mie Gacoan, Bojongsoang Jl. Bojongsoang No. 12 Bandung": "Mie Gacoan, Bojongsoang",
    }
    for raw, expected in merchant_trim_cases.items():
        got, conf, trace = extract_merchant_mbanking(["Transaction Details", "-Rp154.000", "Merchant Name", raw, "Transaction ID"])
        assert got == expected, f"{raw!r}: got={got!r}, expected={expected!r}\n{trace}"
        assert conf >= 0.70, conf

    bad_date_merchant, _, _ = extract_merchant_mbanking([
        "Transaction Details", "-Rp20.000", "21 Mar 2026", "Transaction ID", "260303-DHM8-X6ACYO"
    ])
    assert bad_date_merchant == "Merchant tidak terdeteksi", bad_date_merchant

    bad_id_merchant, _, _ = extract_merchant_mbanking([
        "Transaction Details", "-Rp20.000", "Merchant Name", "260303-DHM8-X6ACYO", "Date and time"
    ])
    assert bad_id_merchant == "Merchant tidak terdeteksi", bad_id_merchant

    amount_no_pan, amount_no_pan_conf, amount_trace = extract_amount_mbanking([
        "Transaction Details",
        "Merchant PAN",
        "936000123456789012",
        "Customer PAN",
        "936000987654321012",
        "Terminal ID",
        "A1234567",
        "Nominal",
        "Rp20.000",
    ])
    assert amount_no_pan == 20000, amount_trace
    assert amount_no_pan_conf >= 0.70, amount_no_pan_conf

    category_cases = {
        "LIQUID LAUNDRY QR": "lainnya",
        "Indomaret": "belanja",
        "Alfamart": "belanja",
        "Mie Gacoan": "makanan_minuman",
        "Kantin TULT-Tenant 3 QR": "makanan_minuman",
        "Mixue, Patriot Jakasampurna": "makanan_minuman",
    }
    for merchant_name, expected_category in category_cases.items():
        got_category, got_conf, _ = _smoke_category(merchant_name, "", merchant_conf=0.80)
        assert got_category == expected_category, f"{merchant_name!r}: {got_category}"
        assert got_conf < 1.0

    receipt_merchant, receipt_conf, _ = extract_merchant_receipt([
        "TOKO ROTI INDAH",
        "Jl. Melati No. 8",
        "ROTI COKLAT 15.000",
        "TOTAL 15.000",
    ])
    assert receipt_merchant == "TOKO ROTI INDAH", receipt_merchant
    assert receipt_conf >= 0.70, receipt_conf

    assert detect_receipt_currency(["TOTAL", "$12.50"]) == "USD"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    run_unit_checks()
    print("Reusable parser unit checks passed.\n")
    run_dataset_smoke(limit_per_group=20)

    PROJECT_ROOT = Path(__file__).resolve().parent
    test_jsonl = PROJECT_ROOT / "data_processed" / "unified" / "test.jsonl"
    if not test_jsonl.exists():
        print(f"Dataset test file not found, skipped OCR sample run: {test_jsonl}")
        return

    # Ambil hanya entry mbanking
    with open(test_jsonl, encoding="utf-8") as f:
        mbanking_entries = [
            json.loads(line) for line in f
            if line.strip() and json.loads(line).get("source") == "mbanking"
        ][:5]   # 5 sampel saja

    print(f"Picked {len(mbanking_entries)} M-Banking samples\n")

    print("Initializing EasyOCR (run pertama download ~64 MB models)…")
    parser = MBankingParser()
    print("Ready.\n")

    for i, entry in enumerate(mbanking_entries, 1):
        print("=" * 72)
        print(f"[{i}/{len(mbanking_entries)}] {Path(entry['image_path']).name}")
        print(f"  GROUND TRUTH:")
        print(f"    amount    = {entry.get('total_amount')}")
        print(f"    date      = {entry.get('transaction_date')!r}")
        print(f"    recipient = {entry.get('merchant')!r}")

        try:
            t0 = time.time()
            result = parser.parse(entry["image_path"], return_raw=True)
            elapsed = time.time() - t0
            print(f"\n  PREDICTION ({elapsed:.1f}s on CPU):")
            print(f"    amount    = {result['amount']}")
            print(f"    date      = {result['date']!r}")
            print(f"    recipient = {result['recipient']!r}")
            print(f"    n_lines   = {result['n_lines']} OCR lines extracted")
            print(f"\n  RAW OCR (preview):")
            preview = result.get("raw_text", "")[:500]
            print(f"    {preview}{'…' if len(preview) >= 500 else ''}")
        except Exception as exc:                    # noqa: BLE001
            print(f"\n  ERROR: {type(exc).__name__}: {exc}")
        print()


if __name__ == "__main__":
    main()
