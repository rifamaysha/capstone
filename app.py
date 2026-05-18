"""Smart Personal Expense — Streamlit application.

Merangkai semua komponen capstone:
  - DONUT (foto struk) / M-Banking parser (screenshot)
  - HybridCategoryClassifier (8 kategori)
  - 50/30/20 budget analyzer
  - Isolation Forest anomaly detection
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Project modules
from donut_inference   import ReceiptParser
from mbanking_inference import MBankingParser
from indobert           import HybridCategoryClassifier
from indobert.categories import CATEGORIES, CATEGORY_DISPLAY
from recommendation     import (
    BUCKET_DISPLAY,
    IDEAL_RATIO,
    analyze_budget,
    detect_anomalies,
)

# --- Suppress noisy library logs ---
for name in ["transformers", "huggingface_hub", "httpx", "easyocr"]:
    logging.getLogger(name).setLevel(logging.ERROR)

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent
DATA_FILE   = PROJECT_ROOT / "data"   / "transactions.json"
TMP_DIR     = PROJECT_ROOT / "tmp"
MODEL_DIR   = PROJECT_ROOT / "models" / "indobert" / "run1" / "final"

# --- Brand colors (Teal Trust palette) ---
PRIMARY  = "#028090"
SECOND   = "#00A896"
ACCENT   = "#02C39A"
WARN     = "#F4845F"
DARK     = "#0F2027"

# ============================================================
# PAGE CONFIG (must be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="Smart Personal Expense",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# MODEL LOADING (cached — load sekali, pakai berulang)
# ============================================================
@st.cache_resource(show_spinner="Loading DONUT model...")
def load_donut() -> ReceiptParser:
    return ReceiptParser()

@st.cache_resource(show_spinner="Loading EasyOCR...")
def load_mbanking() -> MBankingParser:
    return MBankingParser()

@st.cache_resource(show_spinner="Loading IndoBERT hybrid classifier...")
def load_classifier() -> HybridCategoryClassifier:
    return HybridCategoryClassifier(MODEL_DIR)

# ============================================================
# DATA PERSISTENCE — JSON file sederhana
# ============================================================
def load_transactions() -> list[dict[str, Any]]:
    if DATA_FILE.exists():
        with open(DATA_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []

def save_transactions(transactions: list[dict[str, Any]]) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(transactions, f, indent=2, ensure_ascii=False, default=str)

# ============================================================
# HELPERS
# ============================================================
def _parse_amount_str(s: Any) -> float:
    """'72,500' / '1.591.600' / 'Rp 50000' → 50000.0"""
    if s is None:
        return 0.0
    cleaned = "".join(c for c in str(s) if c.isdigit())
    return float(cleaned) if cleaned else 0.0

# ============================================================
# OCR HELPERS untuk receipt (fallback untuk yg DONUT tidak ekstrak)
# ============================================================
_MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "mei": "05", "jun": "06", "jul": "07",
    "aug": "08", "agu": "08", "sep": "09", "okt": "10",
    "oct": "10", "nov": "11", "des": "12", "dec": "12",
}

_RECEIPT_DATE_PATTERNS = [
    re.compile(
        r"(\d{1,2})\s+(Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|"
        r"September|Oktober|November|Desember|Jan|Feb|Mar|Apr|May|Mei|"
        r"Jun|Jul|Aug|Agu|Sep|Sept|Okt|Oct|Nov|Des|Dec)[a-z]*\s+(\d{2,4})",
        re.IGNORECASE,
    ),
    re.compile(r"(\d{1,2})[/\.\-](\d{1,2})[/\.\-](\d{2,4})"),
    re.compile(r"(\d{4})[/\.\-](\d{1,2})[/\.\-](\d{1,2})"),
]

_TOTAL_KEYWORDS = ("grand total", "total", "tunai", "jumlah bayar", "bayar")

# Prefix baris yang BUKAN merchant name (alamat / kontak / dst.)
_MERCHANT_SKIP_PREFIXES = (
    "ruko", "jl.", "jl ", "jalan", "perumahan", "komplek",
    "alamat", "telp", "tel.", "phone", "ph.",
    "no.", "lt.", "lantai", "blok", "kel.", "kec.",
    "kota", "kab.", "(",
)


def _ocr_lines(image_path: Path, reader) -> list[str]:
    """Ambil text lines via EasyOCR (reuse reader dari M-Banking parser)."""
    raw = reader.readtext(str(image_path), detail=0)
    return [str(line).strip() for line in raw if str(line).strip()]


def _is_merchant_line(line: str) -> bool:
    """Apakah line ini calon merchant name (bukan alamat / kontak / nomor)?"""
    stripped = line.strip()
    if len(stripped) < 3:
        return False
    if not re.search(r"[A-Za-z]{3,}", stripped):
        return False
    lower = stripped.lower()
    if any(s in lower for s in ("http", "www.", "@", ".com", ".id", ".co.")):
        return False
    if any(lower.startswith(p) for p in _MERCHANT_SKIP_PREFIXES):
        return False
    # Skip kalau didominasi angka
    letters = sum(c.isalpha() for c in stripped)
    digits = sum(c.isdigit() for c in stripped)
    if digits > letters:
        return False
    return True


def _ocr_extract_merchant(lines: list[str]) -> str:
    """Cari nama merchant — skip alamat/kontak/nomor."""
    # Strict: filter prefix alamat & lainnya
    for line in lines[:10]:
        if _is_merchant_line(line):
            return line.strip()
    # Lenient fallback: baris pertama yang ada hurufnya
    for line in lines[:8]:
        if re.search(r"[A-Za-z]{3,}", line) and len(line.strip()) >= 4:
            return line.strip()
    return ""


def _ocr_extract_date(full_text: str) -> str:
    full_text = _normalize_ocr_digits(full_text)    # NEW
    for pattern in _RECEIPT_DATE_PATTERNS:
        m = pattern.search(full_text)
        if m:
            return m.group(0)
    return ""

def _normalize_ocr_digits(text: str) -> str:
    """Fix common OCR digit confusions in numeric context.
    'Rp38.OOO' → 'Rp38.000', 'RplOS' → 'Rp105', '2O24' → '2024'.
    """
    # 'O'/'o' setelah digit → '0'
    text = re.sub(r"(?<=\d)[Oo]", "0", text)
    # 'O'/'o' sebelum digit → '0'
    text = re.sub(r"[Oo](?=\d)", "0", text)
    # Series 'O' setelah separator (. atau ,) → semua jadi '0'
    text = re.sub(
        r"([.,])([Oo]+)",
        lambda m: m.group(1) + "0" * len(m.group(2)),
        text,
    )
    # 'I'/'l' di antara digit → '1' (konservatif: hanya kalau di-flank digit)
    text = re.sub(r"(?<=\d)[Il](?=\d)", "1", text)
    return text

def _largest_amount_from_text(text: str) -> float:
    """Cari amount terbesar — prioritaskan yang ada prefix 'Rp' (biasanya Total)."""
    text = _normalize_ocr_digits(text)

    # Strategy 1: amount dengan prefix Rp (biasanya total/tunai/payment)
    rp_matches = re.findall(r"rp\.?\s*([\d.,]+)", text.lower())
    rp_valid: list[float] = []
    for m in rp_matches:
        cleaned = "".join(c for c in m if c.isdigit())
        if cleaned and 3 <= len(cleaned) <= 9:
            rp_valid.append(float(cleaned))
    if rp_valid:
        return max(rp_valid)

    # Strategy 2: largest amount tanpa Rp (fallback untuk receipt yang plain)
    thousand_matches = re.findall(r"\b(\d{1,3}(?:[.,]\d{3})+)\b", text)
    valid: list[float] = []
    for m in thousand_matches:
        cleaned = "".join(c for c in m if c.isdigit())
        if cleaned and 3 <= len(cleaned) <= 9:
            valid.append(float(cleaned))
    return max(valid) if valid else 0.0

def _ocr_extract_items(lines: list[str], max_items: int = 10) -> list[str]:
    """Heuristic fallback: extract item-like lines (letters + price-looking digits).

    Dipakai kalau DONUT tidak berhasil ekstrak items proper.
    """
    skip_keywords = (
        "ruko", "jl.", "jl ", "jalan", "alamat", "telp", "tel.",
        "subtotal", "total", "tunai", "payment", "debit", "credit",
        "check no", "closed", "www", ".com", "@", "kasir", "lunas",
        "pos1", "open", "tax", "ppn", "bayar",
    )
    items: list[str] = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) < 5:
            continue
        lower = stripped.lower()
        # Skip header / total / address / footer
        if any(s in lower for s in skip_keywords):
            continue
        # Skip date/time lines
        if re.search(r"\d{1,2}[:.]\d{1,2}[:.]\d{2,4}", stripped):
            continue
        # Harus ada minimal 3 huruf DAN price pattern (3+ digit, biasanya dengan separator)
        has_letters = bool(re.search(r"[A-Za-z]{3,}", stripped))
        has_price = bool(re.search(r"\d{1,3}[.,]\d{3}|\d{4,}", stripped))
        if not (has_letters and has_price):
            continue
        items.append(stripped)
        if len(items) >= max_items:
            break
    return items
    
def _ocr_extract_total(lines: list[str]) -> float:
    """Total amount: try keyword-anchored, fallback to largest amount in text."""
    # Strategy 1: setelah keyword TOTAL/TUNAI/dll
    for i, line in enumerate(lines):
        lower = line.lower()
        if not any(kw in lower for kw in _TOTAL_KEYWORDS):
            continue
        for check in [line] + lines[i + 1 : i + 6]:
            m = re.search(r"(?:rp\.?\s*)?([\d.,]+)", check.lower())
            if m:
                cleaned = "".join(c for c in m.group(1) if c.isdigit())
                if cleaned and 3 <= len(cleaned) <= 9:
                    return float(cleaned)
    # Strategy 2: fallback — largest amount anywhere
    return _largest_amount_from_text(" ".join(lines))

def _filter_donut_items(items: list[str]) -> list[str]:
    """Skip 'items' yang sebenarnya address/contact (DONUT salah klasifikasi)."""
    skip_prefixes = (
        "jl.", "jl ", "jalan", "ruko", "perumahan", "komplek",
        "alamat", "telp", "tel.", "lt.", "lantai", "blok",
        "kel.", "kec.",
    )
    skip_contains = ("www", ".com", "@", "http")
    cleaned: list[str] = []
    for item in items:
        if not item or not item.strip():
            continue
        lower = item.lower().strip()
        if any(lower.startswith(p) for p in skip_prefixes):
            continue
        if any(s in lower for s in skip_contains):
            continue
        cleaned.append(item)
    return cleaned

def _expand_year(year: str) -> str:
    """'19' → '2019', '99' → '1999', '2024' → '2024'."""
    if len(year) == 4:
        return year
    if len(year) == 2:
        y = int(year)
        return f"20{year}" if y <= 30 else f"19{year}"
    return year


def _normalize_date(date_str: str) -> str:
    """Normalize ke format DD/MM/YYYY."""
    if not date_str or not date_str.strip():
        return ""
    s = date_str.strip()

    # "13 May 2026" / "13 Mei 26"
    m = re.match(r"(\d{1,2})\s+([A-Za-z]+)\.?\s+(\d{2,4})", s)
    if m:
        day, mon, year = m.group(1).zfill(2), m.group(2)[:3].lower(), m.group(3)
        mn = _MONTH_MAP.get(mon)
        if mn:
            return f"{day}/{mn}/{_expand_year(year)}"

    # "May 13, 2026"
    m = re.match(r"([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{2,4})", s)
    if m:
        mon, day, year = m.group(1)[:3].lower(), m.group(2).zfill(2), m.group(3)
        mn = _MONTH_MAP.get(mon)
        if mn:
            return f"{day}/{mn}/{_expand_year(year)}"

    # "18.7.2024" / "18/7/2024" / "18-7-24"
    m = re.match(r"(\d{1,2})[/\.\-](\d{1,2})[/\.\-](\d{2,4})", s)
    if m:
        d, mo, y = m.group(1).zfill(2), m.group(2).zfill(2), m.group(3)
        return f"{d}/{mo}/{_expand_year(y)}"

    # ISO "2024-07-18"
    m = re.match(r"(\d{4})[/\.\-](\d{1,2})[/\.\-](\d{1,2})", s)
    if m:
        y, mo, d = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
        return f"{d}/{mo}/{y}"

    return s   # fallback: return as-is
    
def process_uploaded_image(
    image_path: Path,
    is_screenshot: bool,
    donut: ReceiptParser,
    mbanking: MBankingParser,
    classifier: HybridCategoryClassifier,
) -> dict[str, Any]:
    """Pipeline lengkap: gambar → ekstraksi → klasifikasi → dict."""
    raw_ocr_text = ""

    if is_screenshot:
        ext = mbanking.parse(image_path, return_raw=True)
        merchant = ext.get("recipient") or ""
        amount   = float(ext.get("amount") or 0)
        date     = ext.get("date") or ""
        items: list[str] = []
        raw_ocr_text = ext.get("raw_text") or ""
        text_for_class = merchant or raw_ocr_text[:200]
        method = "M-Banking parser"

        # Fallback amount: OCR sendiri kalau parser miss
        if amount == 0:
            sc_lines = _ocr_lines(image_path, mbanking.reader)
            raw_ocr_text = "\n".join(sc_lines)
            amount = _largest_amount_from_text(raw_ocr_text)
    else:
        # DONUT untuk items + total
        ext = donut.parse(image_path)
        menu = ext.get("menu", [])
        if isinstance(menu, dict):
            menu = [menu]
        raw_items = [m.get("nm", "") for m in menu
                     if isinstance(m, dict) and m.get("nm")]
        items = _filter_donut_items(raw_items)

        total_data = ext.get("total", {})
        amount_donut = _parse_amount_str(
            total_data.get("total_price") if isinstance(total_data, dict) else None
        )

        ocr_lines = _ocr_lines(image_path, mbanking.reader)
        raw_ocr_text = "\n".join(ocr_lines)

        # NEW: OCR fallback items kalau DONUT tidak proper
        if not items:
            items = _ocr_extract_items(ocr_lines)

        merchant = _ocr_extract_merchant(ocr_lines)
        date     = _ocr_extract_date(raw_ocr_text)
        amount   = amount_donut if amount_donut > 0 else _ocr_extract_total(ocr_lines)

        text_for_class = " ".join(filter(None, [merchant] + items))
        method = "DONUT + OCR"

    # Normalize date ke DD/MM/YYYY
    date = _normalize_date(date)

    # NEW: strip trailing punctuation dari merchant (logo OCR sering tambah ", ®, dst.)
    merchant = merchant.strip(' "\'`,.;:®()') if merchant else ""

    cls = (classifier.predict(text_for_class)
           if text_for_class.strip()
           else {"label": "lainnya", "confidence": 0.0, "source": "empty"})

    return {
        "method":                method,
        "merchant":              merchant,
        "amount":                amount,
        "date":                  date,
        "items":                 items,
        "category":              cls["label"],
        "confidence":            cls["confidence"],
        "classification_source": cls.get("source", "?"),
        "raw_ocr":               raw_ocr_text,   # NEW: untuk debug expander
    }
    
# ============================================================
# HEADER
# ============================================================
st.markdown(
    f"""
    <div style="padding:1.5rem 0 1rem 0;border-bottom:3px solid {PRIMARY};margin-bottom:1.5rem;">
      <h1 style="margin:0;color:{DARK};">💰 Smart Personal Expense</h1>
      <p style="margin:0.3rem 0 0;color:#64748B;font-style:italic;">
        Pencatatan pengeluaran otomatis dari foto struk & screenshot M-Banking
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# TABS
# ============================================================
tab_upload, tab_history, tab_dashboard = st.tabs([
    "📷 Upload Transaksi",
    "📋 Riwayat",
    "📊 Dashboard & Saran",
])

# ============================================================
# TAB 1 — UPLOAD
# ============================================================
with tab_upload:
    st.subheader("Tambah Transaksi Baru")

    src = st.radio(
        "Jenis gambar yang akan diunggah:",
        ["Foto Struk Belanja", "Screenshot M-Banking"],
        horizontal=True,
    )

    uploaded = st.file_uploader(
        "Pilih gambar (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded:
        # Save temporarily
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = TMP_DIR / uploaded.name
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getvalue())

        col_img, col_form = st.columns([1, 1.2])

        with col_img:
            st.image(uploaded, use_container_width=True, caption=uploaded.name)

        with col_form:
            is_screenshot = src == "Screenshot M-Banking"
            with st.spinner("Mengekstrak data — ini bisa 30 detik di CPU..."):
                donut = load_donut()
                mbank = load_mbanking()
                classifier = load_classifier()
                result = process_uploaded_image(
                    tmp_path, is_screenshot, donut, mbank, classifier
                )

            st.success(
                f"✓ Ekstraksi selesai via **{result['method']}**  ·  "
                f"Kategori prediksi: **{CATEGORY_DISPLAY.get(result['category'], result['category'])}** "
                f"({result['confidence']:.0%} dari {result['classification_source']})"
            )

            # NEW: debug — lihat apa yang OCR benar-benar baca
            with st.expander("🔍 Debug: Raw OCR Text (untuk troubleshoot)"):
                st.text(result.get("raw_ocr", "(tidak tersedia)") or "(kosong)")

            st.markdown("### Verifikasi & Edit")


            merchant = st.text_input(
                "Merchant",
                value=result["merchant"],
                placeholder="Nama toko / penerima",
            )
            col_a, col_d = st.columns(2)
            with col_a:
                amount = st.number_input(
                    "Amount (Rp)",
                    value=float(result["amount"]),
                    step=1000.0,
                    format="%.0f",
                )
            with col_d:
                date = st.text_input(
                    "Tanggal",
                    value=result["date"],
                    placeholder="DD/MM/YYYY",
                )

            # Category selector with prediction as default
            cat_keys = list(CATEGORIES)
            try:
                default_idx = cat_keys.index(result["category"])
            except ValueError:
                default_idx = cat_keys.index("lainnya")

            category = st.selectbox(
                "Kategori",
                options=cat_keys,
                format_func=lambda x: CATEGORY_DISPLAY.get(x, x),
                index=default_idx,
            )

            if result["items"]:
                with st.expander(f"Items terdeteksi ({len(result['items'])})"):
                    for it in result["items"]:
                        st.write(f"• {it}")

            if st.button("💾 Simpan Transaksi", type="primary", use_container_width=True):
                if amount <= 0:
                    st.error("Amount harus > 0")
                else:
                    txs = load_transactions()
                    new_tx = {
                        "id":         len(txs) + 1,
                        "date":       date or datetime.now().strftime("%d %b %Y"),
                        "merchant":   merchant or "(tanpa nama)",
                        "amount":     float(amount),
                        "category":   category,
                        "source":     "screenshot" if is_screenshot else "receipt",
                        "saved_at":   datetime.now().isoformat(timespec="seconds"),
                    }
                    txs.append(new_tx)
                    save_transactions(txs)
                    st.success(f"Tersimpan! Total transaksi: {len(txs)}")
                    st.balloons()

# ============================================================
# TAB 2 — HISTORY
# ============================================================
with tab_history:
    st.subheader("Riwayat Transaksi")
    txs = load_transactions()

    if not txs:
        st.info("Belum ada transaksi. Upload gambar di tab pertama.")
    else:
        df = pd.DataFrame(txs)
        df["Kategori"] = df["category"].map(CATEGORY_DISPLAY)
        df["Amount (Rp)"] = df["amount"].apply(lambda x: f"Rp {x:,.0f}")
        df_display = df[["date", "merchant", "Amount (Rp)", "Kategori", "source"]]
        df_display.columns = ["Tanggal", "Merchant", "Amount (Rp)", "Kategori", "Sumber"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transaksi",   len(txs))
        col2.metric("Total Pengeluaran", f"Rp {df['amount'].sum():,.0f}")
        col3.metric("Rata-rata",         f"Rp {df['amount'].mean():,.0f}")

        st.dataframe(df_display, use_container_width=True, hide_index=True)

        if st.button("🗑️ Hapus Semua (demo reset)", type="secondary"):
            save_transactions([])
            st.rerun()

# ============================================================
# TAB 3 — DASHBOARD
# ============================================================
with tab_dashboard:
    st.subheader("Analisa Pengeluaran")
    txs = load_transactions()

    if not txs:
        st.info("Belum ada data. Upload transaksi dulu di tab pertama.")
    else:
        # Income input
        income = st.number_input(
            "💵 Pendapatan bulanan kamu (Rp)",
            min_value=100_000,
            value=5_000_000,
            step=500_000,
            format="%.0f",
        )

        result = analyze_budget(txs, income)

        # ---- Top metrics ----
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pendapatan",     f"Rp {result['income']:,.0f}")
        c2.metric("Pengeluaran",    f"Rp {result['total_spent']:,.0f}")
        c3.metric("Tabungan",       f"Rp {result['tabungan']:,.0f}")
        c4.metric("Rasio Tabungan", f"{result['actual_ratio']['tabungan']:.0%}",
                  delta=f"{(result['actual_ratio']['tabungan'] - 0.20)*100:+.0f}pp vs target",
                  delta_color="normal" if result['actual_ratio']['tabungan'] >= 0.20 else "inverse")

        st.divider()

        # ---- Charts row ----
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("##### Aktual vs Ideal — Aturan 50/30/20")
            buckets = ["kebutuhan", "keinginan", "tabungan"]
            fig = go.Figure(data=[
                go.Bar(
                    name="Aktual",
                    x=[BUCKET_DISPLAY[b] for b in buckets],
                    y=[result["actual_ratio"][b] * 100 for b in buckets],
                    marker_color=PRIMARY,
                    text=[f"{result['actual_ratio'][b]*100:.0f}%" for b in buckets],
                    textposition="outside",
                ),
                go.Bar(
                    name="Ideal",
                    x=[BUCKET_DISPLAY[b] for b in buckets],
                    y=[IDEAL_RATIO[b] * 100 for b in buckets],
                    marker_color=ACCENT,
                    text=[f"{IDEAL_RATIO[b]*100:.0f}%" for b in buckets],
                    textposition="outside",
                ),
            ])
            fig.update_layout(
                barmode="group",
                yaxis_title="% dari pendapatan",
                height=350,
                margin=dict(l=20, r=20, t=30, b=20),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown("##### Distribusi per Kategori")
            cats_df = pd.DataFrame([
                {"Kategori": CATEGORY_DISPLAY.get(k, k), "Total": v}
                for k, v in result["by_category"].items()
                if v > 0
            ]).sort_values("Total", ascending=False)

            fig = px.pie(
                cats_df, names="Kategori", values="Total", hole=0.45,
                color_discrete_sequence=[
                    PRIMARY, SECOND, ACCENT, WARN,
                    "#5BC0BE", "#3A506B", "#9DCEFF", "#94A3B8",
                ],
            )
            fig.update_traces(textposition="outside", textinfo="label+percent")
            fig.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=30, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ---- Recommendations ----
        st.markdown("##### 💡 Rekomendasi")
        for rec in result["recommendations"]:
            if "Pertahankan" in rec:
                st.success(rec)
            elif "Over" in rec or "OVER" in rec:
                st.warning(rec)
            else:
                st.info(rec)

        st.divider()

        # ---- Anomaly Detection ----
        st.markdown("##### 🚨 Anomaly Detection — Isolation Forest")
        anomalies = detect_anomalies(txs, contamination=0.1, by_category=True)
        if anomalies:
            df_anom = pd.DataFrame([
                {
                    "Tanggal":  a.get("date", "-"),
                    "Merchant": a.get("merchant", "-"),
                    "Amount":   f"Rp {a['amount']:,.0f}",
                    "Kategori": CATEGORY_DISPLAY.get(a["category"], a["category"]),
                    "Score":    f"{a['anomaly_score']:.3f}",
                }
                for a in sorted(anomalies, key=lambda x: x["anomaly_score"])
            ])
            st.dataframe(df_anom, use_container_width=True, hide_index=True)
            st.caption("Score makin negatif = makin tidak biasa dibanding pola normal kategori ini.")
        else:
            st.success("Tidak ada transaksi outlier terdeteksi. Pola pengeluaran konsisten 👍")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    f"Smart Personal Expense  ·  Capstone Kelompok 24  ·  "
    f"Data tersimpan lokal di `{DATA_FILE.name}`"
)