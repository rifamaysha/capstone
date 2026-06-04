# Smart Personal Expense

Smart Personal Expense adalah aplikasi pencatatan pengeluaran pribadi dari struk belanja dan screenshot pembayaran. Frontend utama memakai React, backend utama memakai FastAPI, dan OCR/parser dipakai untuk membantu membaca merchant, nominal, tanggal, kategori, serta jenis transaksi.

## Tech Stack

- FastAPI untuk backend API
- React + Vite untuk frontend
- EasyOCR untuk OCR cepat saat demo
- Parser OCR custom untuk struk belanja dan screenshot pembayaran
- IndoBERT hybrid classifier untuk kategori jika model tersedia
- JSON lokal sebagai storage transaksi development

## Struktur Utama

```text
backend/                       FastAPI app dan service layer
frontend/                      React + Vite app
assets/                        Logo dan asset UI
tools/                         Diagnostic/evaluation tools
extraction_postprocessor.py    Postprocess OCR field extraction
mbanking_inference.py          OCR/parser screenshot pembayaran
donut_inference.py             Parser DONUT legacy/fallback
indobert/hybrid.py             Hybrid category classifier
recommendation/anomaly.py      Insight dan anomaly helper
```

`app.py` adalah legacy Streamlit prototype untuk backup lokal. Entry point utama project ini adalah FastAPI + React.

## Menjalankan Backend

```powershell
cd "C:\Users\ASUS\Documents\capstone(2)\capstone-main"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Backend berjalan di:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

## Menjalankan Frontend

```powershell
cd "C:\Users\ASUS\Documents\capstone(2)\capstone-main\frontend"
npm install
npm run dev
```

Frontend berjalan di:

```text
http://localhost:5173
```

## Catatan Model

Folder `models/` tidak dicommit karena ukuran file model besar. File model IndoBERT perlu didownload terpisah dan diletakkan di:

```text
models/indobert/run1/final/model.safetensors
```

Untuk distribusi model, gunakan Google Drive, Hugging Face, atau GitHub Release, lalu letakkan file sesuai path di atas.

## Catatan Dataset dan Data Lokal

Folder `dataset/` tidak dicommit karena berisi dataset lokal/besar.

File transaksi lokal juga tidak dicommit:

```text
data/transactions.json
```

File ini akan dibuat/dipakai saat aplikasi berjalan untuk menyimpan transaksi development.

## Upload dan Batch

Halaman Upload & Proses mendukung:

- Scan transaksi dari struk belanja atau screenshot pembayaran
- Input manual transaksi tanpa OCR
- Batch upload maksimal 3 transaksi
- Batch diproses sequential satu per satu, bukan paralel
- User tetap harus mengecek dan menyimpan transaksi, tidak auto-save

## API Ringkas

| Method | Path | Fungsi |
| --- | --- | --- |
| GET | `/health` | Cek status backend |
| POST | `/extract` | Upload gambar dan ekstrak transaksi |
| GET | `/transactions` | Ambil semua transaksi |
| POST | `/transactions` | Simpan transaksi |
| DELETE | `/transactions` | Hapus semua transaksi lokal |
| GET | `/insights` | Ambil insight pengeluaran |

## Catatan Pengembangan

- React adalah frontend utama.
- FastAPI adalah backend utama.
- Streamlit `app.py` hanya legacy prototype lokal.
- Jangan commit `venv/`, `tmp/`, `dataset/`, `models/`, `frontend/node_modules/`, `frontend/dist/`, atau data transaksi lokal.
- Untuk evaluasi parser, gunakan:

```powershell
python tools/parser_evaluation.py --limit 50
```
