"""Convert UnifiedRecord → DONUT target string (XML-like tags)."""
from __future__ import annotations

import xml.sax.saxutils as saxutils

from .schema import TransactionItem, UnifiedRecord


def _esc(s: str) -> str:
    """XML-escape karakter spesial (&, <, >, ', \")."""
    return saxutils.escape(s, {'"': "&quot;", "'": "&apos;"})


def _fmt_amount(amount: float | None) -> str | None:
    """Format angka: integer kalau bulat, 2 desimal kalau ada pecahan."""
    if amount is None:
        return None
    return str(int(amount)) if amount == int(amount) else f"{amount:.2f}"


def _encode_item(item: TransactionItem) -> str:
    """Encode satu item dalam tag <s_item>...</s_item>."""
    parts = [f"<s_name>{_esc(item.name)}</s_name>"]
    if item.quantity is not None:
        parts.append(f"<s_qty>{_fmt_amount(item.quantity)}</s_qty>")
    if item.unit_price is not None:
        parts.append(f"<s_unit_price>{_fmt_amount(item.unit_price)}</s_unit_price>")
    if item.total_price is not None:
        parts.append(f"<s_price>{_fmt_amount(item.total_price)}</s_price>")
    return f"<s_item>{''.join(parts)}</s_item>"


def encode_donut_target(record: UnifiedRecord) -> str:
    """Encode UnifiedRecord menjadi string target DONUT.

    Format: tag XML-like. Hanya field yang ada nilainya yang di-include
    (model tidak perlu belajar memprediksi tag kosong → target lebih ringkas).

    Args:
        record: UnifiedRecord hasil parsing salah satu loader.

    Returns:
        String target untuk training DONUT, contoh:
        '<s_source>cord</s_source><s_total>1591600</s_total>
         <s_currency>IDR</s_currency><s_items><s_item>...</s_item></s_items>'
    """
    parts = [f"<s_source>{record.source}</s_source>"]

    if record.merchant:
        parts.append(f"<s_merchant>{_esc(record.merchant)}</s_merchant>")
    if record.transaction_date:
        parts.append(f"<s_date>{_esc(record.transaction_date)}</s_date>")

    total_str = _fmt_amount(record.total_amount)
    if total_str:
        parts.append(f"<s_total>{total_str}</s_total>")

    if record.currency:
        parts.append(f"<s_currency>{record.currency}</s_currency>")
    if record.category:
        parts.append(f"<s_category>{_esc(record.category)}</s_category>")

    if record.items:
        items_xml = "".join(_encode_item(it) for it in record.items)
        parts.append(f"<s_items>{items_xml}</s_items>")

    return "".join(parts)


def collect_special_tokens() -> list[str]:
    """Daftar semua special token yang dipakai target string.

    Dipanggil di Bagian 3 saat `tokenizer.add_special_tokens(...)`
    sebelum fine-tune DONUT.
    """
    return [
        "<s_source>", "</s_source>",
        "<s_merchant>", "</s_merchant>",
        "<s_date>", "</s_date>",
        "<s_total>", "</s_total>",
        "<s_currency>", "</s_currency>",
        "<s_category>", "</s_category>",
        "<s_items>", "</s_items>",
        "<s_item>", "</s_item>",
        "<s_name>", "</s_name>",
        "<s_qty>", "</s_qty>",
        "<s_unit_price>", "</s_unit_price>",
        "<s_price>", "</s_price>",
    ]