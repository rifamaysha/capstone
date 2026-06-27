# Smart Personal Expense

> **Capstone Project — Kelompok 24 | Universitas Telkom 2026**  
> Sistem pencatatan dan kategorisasi pengeluaran otomatis dari foto struk & screenshot M-Banking

---

## Tentang Proyek

**Smart Personal Expense** adalah aplikasi manajemen keuangan pribadi berbasis AI yang memungkinkan pengguna mencatat pengeluaran secara otomatis—tanpa input manual—hanya dengan mengunggah foto struk belanja atau screenshot pembayaran digital.

Sistem ini dikembangkan sebagai respons terhadap rendahnya indeks literasi keuangan Indonesia (65,43%, SNLIK 2024). Pipeline-nya menggabungkan tiga teknologi utama:

| Komponen | Teknologi | Fungsi |
|---|---|---|
| OCR Struk | DONUT (`donut-base-finetuned-cord-v2`) | Ekstraksi merchant, nominal, tanggal, item |
| OCR M-Banking | EasyOCR + rule-based parser | Parsing screenshot GoPay, OVO, DANA, BCA, Mandiri |
| Klasifikasi | IndoBERT Hybrid | Kategorisasi ke 8 kategori pengeluaran |
| Anomali | Isolation Forest | Deteksi transaksi tidak wajar |
| Rekomendasi | Rule-based 50/30/20 | Saran penghematan personal |

---

## Struktur Repositori

```
capstone/
│
├── app.py                      # Entry point — aplikasi Streamlit utama
├── donut_inference.py          # ReceiptParser: wrapper DONUT untuk struk fisik
├── mbanking_inference.py       # MBankingParser: EasyOCR + regex untuk screenshot
│
├── train_donut.py              # Script fine-tuning DONUT
├── train_indobert.py           # Script fine-tuning IndoBERT
├── donut_finetune.zip          # Hasil fine-tuning DONUT (checkpoint)
├── indobert.zip                # Hasil fine-tuning IndoBERT (checkpoint)
│
├── auto_label_dataset.py       # Auto-labeling dataset untuk training
├── build_dataset.py            # Pembangunan dataset training
├── inspect_labels.py           # Inspeksi dan verifikasi label dataset
│
├── pipeline_test.py            # Pengujian pipeline end-to-end
├── test_inference.py           # Unit test DONUT inference
├── test_indobert_inference.py  # Unit test IndoBERT inference
├── test_mbanking.py            # Unit test M-Banking parser
├── test_recommendation.py      # Unit test modul rekomendasi
├── test_loader.py              # Unit test data loader
│
├── recommendation.zip          # Modul analisis budget & deteksi anomali
├── loaders.zip                 # Data loaders untuk training
├── models.zip                  # Model weights (alternatif dari zip indobert)
├── tmp.zip                     # Direktori temporary (auto-generated)
└── __pycache__.zip             # Cache Python (auto-generated)
```

> **Catatan:** File `.zip` berisi modul-modul yang perlu diekstrak sebelum menjalankan aplikasi. Lihat bagian [Instalasi](#-instalasi) di bawah.

---

## Instalasi

### Prasyarat

| Kebutuhan | Versi |
|---|---|
| Python | 3.9 atau lebih baru |
| RAM | Minimal 4 GB (disarankan 8 GB untuk IndoBERT) |
| OS | Windows 10/11, macOS, atau Linux |
| GPU | Opsional (CPU didukung, inferensi lebih lambat) |

### Langkah Instalasi

**1. Clone repositori**

```bash
git clone https://github.com/rifamaysha/capstone.git
cd capstone
```

**2. Buat dan aktifkan virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

**3. Ekstrak modul-modul zip**

```bash
# Ekstrak semua modul yang diperlukan
unzip indobert.zip
unzip recommendation.zip
unzip loaders.zip
unzip models.zip
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

> Jika belum ada `requirements.txt`, install manual:
> ```bash
> pip install streamlit torch transformers easyocr pandas plotly scikit-learn pillow
> ```

**5. Jalankan aplikasi**

```bash
streamlit run app.py
```

Aplikasi akan berjalan di **http://localhost:8501**

---

## Cara Penggunaan

Aplikasi memiliki tiga tab utama yang dapat diakses langsung dari browser:

### Tab 1 — Upload Transaksi

1. Pilih jenis gambar: **Foto Struk Belanja** atau **Screenshot M-Banking**
2. Unggah gambar (PNG/JPG, maks. 10 MB)
3. Tunggu proses OCR (~15–35 detik di CPU)
4. Verifikasi hasil ekstraksi: merchant, nominal, tanggal, kategori
5. Klik **Simpan Transaksi**

### Tab 2 — Riwayat

- Lihat semua transaksi yang tersimpan dalam format tabel
- Ringkasan metrik: total transaksi, total pengeluaran, rata-rata
- Hapus data jika diperlukan (tombol reset tersedia)

### Tab 3 — Dashboard & Saran

- Masukkan **pendapatan bulanan** untuk analisis 50/30/20
- Lihat perbandingan aktual vs. target anggaran per bucket:
  - Kebutuhan (50%)
  - Keinginan (30%)
  - Tabungan (20%)
- Distribusi pengeluaran per kategori (pie chart)
- **Deteksi anomali** Isolation Forest per kategori
- Rekomendasi penghematan otomatis berdasarkan data aktual

---

## Arsitektur Pipeline

```
Gambar Input (struk / screenshot)
         │
         ▼
┌─────────────────────────────┐
│   Deteksi Jenis Dokumen     │
│  (is_screenshot flag)       │
└──────┬──────────────────────┘
       │
       ├─── Struk Fisik ──────► DONUT Inference
       │                        (donut_inference.py)
       │                              │
       │                              ▼
       │                       OCR Fallback jika perlu
       │                       (EasyOCR via mbanking reader)
       │
       └─── Screenshot ────────► EasyOCR + Rule-Based Parser
                                  (mbanking_inference.py)
                                        │
                                        ▼
                              ┌─────────────────────┐
                              │  Post-processing:   │
                              │  - Normalisasi OCR  │
                              │    digits (O→0)     │
                              │  - Parse tanggal    │
                              │  - Normalize amount │
                              └─────────┬───────────┘
                                        │
                                        ▼
                              ┌──────────────────────┐
                              │  HybridClassifier    │
                              │  IndoBERT (primary)  │
                              │  Rule-based (fallback│
                              │  jika conf < 0.6)    │
                              └─────────┬────────────┘
                                        │
                                        ▼
                              Simpan ke data/transactions.json
                                        │
                                        ▼
                              ┌──────────────────────┐
                              │  Analisis & Insight  │
                              │  - Budget 50/30/20   │
                              │  - Isolation Forest  │
                              │  - Rekomendasi       │
                              └──────────────────────┘
```

---

## Kategori Pengeluaran

Sistem mengklasifikasikan transaksi ke dalam **8 kategori**:

| Kode | Tampilan | Contoh |
|---|---|---|
| `makanan_minuman` | Makanan & Minuman | McD, KFC, Warteg, Kopi Kenangan |
| `belanja` | Belanja & Retail | Indomaret, Alfamart, Tokopedia |
| `transportasi` | Transportasi | Grab, Gojek, Bensin, KRL |
| `tagihan` | Tagihan & Utilitas | Token PLN, Pulsa, YouTube, Netflix |
| `kesehatan` | Kesehatan | Apotek, Klinik, Halodoc |
| `hiburan` | Hiburan | Bioskop, konser, game |
| `pendidikan` | Pendidikan | Kursus, buku, alat tulis |
| `lainnya` | Lainnya | Transaksi yang tidak terkategori |

---

## Menjalankan Tests

```bash
# Test pipeline end-to-end
python pipeline_test.py

# Test komponen individual
python test_inference.py        # DONUT
python test_indobert_inference.py  # IndoBERT
python test_mbanking.py         # M-Banking parser
python test_recommendation.py   # Modul rekomendasi
python test_loader.py           # Data loader
```

---

## Training Model (Opsional)

Jika ingin melatih ulang model dari awal:

```bash
# Fine-tune DONUT
python train_donut.py

# Fine-tune IndoBERT
python train_indobert.py

# Build & label dataset
python build_dataset.py
python auto_label_dataset.py
python inspect_labels.py
```

> **Peringatan:** Training DONUT memerlukan GPU dengan VRAM ≥ 12 GB dan waktu ~8 jam. Training di CPU tidak disarankan.

---

## Troubleshooting

| Masalah | Penyebab | Solusi |
|---|---|---|
| `ModuleNotFoundError: indobert` | Modul belum diekstrak | `unzip indobert.zip` |
| `ModuleNotFoundError: recommendation` | Modul belum diekstrak | `unzip recommendation.zip` |
| OCR hasil kosong | Kualitas gambar buruk | Gunakan gambar dengan pencahayaan cukup & resolusi min. 300px |
| Nominal terbaca salah | Struk buram/miring | Crop gambar agar hanya area struk, hindari bayangan |
| Streamlit tidak terbuka | Port 8501 bentrok | `streamlit run app.py --server.port 8502` |
| IndoBERT OOM | RAM tidak cukup | Tutup aplikasi lain, pastikan RAM ≥ 8 GB |
| Download model DONUT lambat | Koneksi internet | Model ~700 MB, diperlukan hanya di run pertama |

---

## Hasil Evaluasi

| Metrik | Nilai |
|---|---|
| CER Ekstraksi Struk (keseluruhan) | **4,7%** |
| Akurasi Klasifikasi IndoBERT | **87,3%** |
| F1-Score Makro | **0,851** |
| Akurasi Parser M-Banking | **93,4%** |
| Presisi Deteksi Anomali | **87,5%** |
| SUS Score (Usability) | **81,4 / 100** |

---

## Tim Pengembang

| Nama | NIM | Peran Utama |
|---|---|---|
| Rifa Mayshakori | 103052300087 | ML Pipeline, Evaluasi Model |
| Rifki Arif | 103052300041 | Backend, Training Model |
| Rini Anisa | 103052300017 | Frontend/UI, User Testing |

**Dosen Pendamping:** Ghina Khoerunnisa, S.Kom., M.Kom  
**Universitas Telkom — Program Studi Sains Data — 2026**

---

## Lisensi

Proyek ini dikembangkan untuk keperluan akademik (Capstone Project Universitas Telkom). Tidak diperuntukkan untuk penggunaan komersial.
