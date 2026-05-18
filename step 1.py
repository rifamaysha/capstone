"""Image preprocessing pipeline untuk Smart Personal Expense.

Mendukung 3 sumber data: struk retail Inggris (Kaggle), struk Indonesia
(CORD v2), dan screenshot M-Banking. Output siap dijadikan input model
DONUT (gambar 3-channel BGR, lebar ter-normalisasi).
"""
#!pip install numpy opencv-python pillow
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from tqdm import tqdm

# ---------------------------------------------------------------------------
# ASUMSI:
# 1. Semua sumber data menghasilkan gambar yang readable oleh cv2.imread
#    (PNG dari Kaggle/CORD, JPEG dari M-Banking). Format lain ditolak eksplisit.
# 2. DONUT mengonsumsi gambar BGR/RGB 3-channel. Kita TIDAK grayscale di akhir
#    pipeline — denoising & CLAHE dilakukan tetapi color channels dipertahankan
#    karena DONUT image processor butuh 3-channel input.
# 3. Target lebar 1200 px adalah kompromi: cukup tajam untuk OCR struk panjang
#    tanpa membebani GPU memory saat batch encoding DONUT.
# 4. Untuk screenshot M-Banking yang sudah lurus, deskew di-skip otomatis
#    karena threshold 0.5° tidak akan terlampaui (atau caller set deskew=False).
# 5. Logging diset di module-level; root config (handler, format) ditangani
#    oleh `src/utils/logging_config.py`.
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# Format yang dijamin didukung OpenCV across platform
_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".bmp", ".tiff", ".tif", ".webp"}

# Ambang sudut deskew: di bawah ini rotasi tidak signifikan & justru dapat
# memperkenalkan blur dari interpolasi cv2.warpAffine.
_DESKEW_MIN_ANGLE_DEG: float = 0.5


def load_image(image_path: str) -> np.ndarray:
    """Load gambar dari disk sebagai array numpy BGR.

    Args:
        image_path: Path absolut/relatif ke file gambar.

    Returns:
        Array numpy dengan shape (H, W, 3) dan dtype uint8 dalam ruang warna BGR.

    Raises:
        FileNotFoundError: Jika file tidak ada di path tersebut.
        ValueError: Jika ekstensi file tidak didukung.
        IOError: Jika file ada tetapi gagal di-decode (kemungkinan corrupt).
    """
    path = Path(image_path)

    if not path.exists():
        # Gagal cepat: lebih informatif daripada error generik dari OpenCV
        raise FileNotFoundError(f"Image not found: {image_path}")

    if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        # Tolak dini agar tidak mengonsumsi memory untuk decode format aneh
        raise ValueError(
            f"Unsupported image format '{path.suffix}'. "
            f"Supported: {sorted(_SUPPORTED_EXTENSIONS)}"
        )

    # cv2.imread tidak raise pada file corrupt — ia mengembalikan None.
    # Jadi kita harus cek manual.
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(
            f"Failed to decode image (corrupt or unreadable): {image_path}"
        )

    logger.debug("Loaded image %s with shape %s", image_path, image.shape)
    return image


def normalize_resolution(
    image: np.ndarray, target_width: int = 1200
) -> np.ndarray:
    """Resize gambar ke lebar target dengan menjaga aspect ratio.

    Args:
        image: Array gambar BGR.
        target_width: Lebar output dalam piksel. Default 1200 mengikuti
            rekomendasi DONUT untuk struk panjang.

    Returns:
        Array gambar yang telah di-resize.

    Raises:
        ValueError: Jika `target_width` <= 0 atau gambar input kosong.
    """
    if target_width <= 0:
        raise ValueError(f"target_width must be positive, got {target_width}")
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")

    h, w = image.shape[:2]

    # Skip resize jika sudah pas — hindari interpolasi yang tidak perlu
    if w == target_width:
        return image

    scale = target_width / w
    target_height = max(1, int(round(h * scale)))

    # Pakai INTER_AREA saat downscale (lebih tajam untuk teks),
    # INTER_CUBIC saat upscale (smoother daripada LINEAR).
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC

    resized = cv2.resize(
        image, (target_width, target_height), interpolation=interpolation
    )
    logger.debug(
        "Resized from (%d, %d) to (%d, %d)", h, w, target_height, target_width
    )
    return resized


def deskew_image(image: np.ndarray) -> np.ndarray:
    """Deteksi & koreksi kemiringan teks pada gambar.

    Menggunakan binarisasi Otsu + minAreaRect pada koordinat piksel teks
    untuk memperkirakan sudut. Jika kemiringan absolut lebih kecil dari
    `_DESKEW_MIN_ANGLE_DEG`, gambar dikembalikan tanpa modifikasi (menghindari
    blur akibat warpAffine yang tidak perlu).

    Args:
        image: Array gambar BGR.

    Returns:
        Gambar yang telah diluruskan (atau gambar asli bila sudut < ambang).
    """
    # Konversi ke grayscale: deskew bekerja pada distribusi spasial piksel teks,
    # informasi warna tidak relevan di sini.
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image.copy()
    )

    # Otsu + invert: teks (gelap di latar terang) jadi piksel putih (foreground).
    # Foreground ini yang akan kita ukur orientasinya.
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    # Kumpulkan koordinat semua piksel teks. Bila kosong, gambar mungkin blank.
    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        logger.warning("No foreground pixels detected; skipping deskew")
        return image

    # minAreaRect mengembalikan ((cx, cy), (w, h), angle) di mana angle ∈ [-90, 0).
    angle = cv2.minAreaRect(coords)[-1]

    # Konvensi OpenCV: normalisasi agar angle merepresentasikan rotasi yang
    # diperlukan untuk meluruskan teks (positive = counter-clockwise).
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < _DESKEW_MIN_ANGLE_DEG:
        logger.info(
            "Skew angle %.3f° below threshold %.2f°; skipping rotation",
            angle,
            _DESKEW_MIN_ANGLE_DEG,
        )
        return image

    # Rotasi terhadap titik pusat agar tidak ada konten yang terpotong jauh
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    # BORDER_REPLICATE: padding tepi diisi piksel terakhir, mencegah strip hitam
    # yang bisa membingungkan model OCR.
    deskewed = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    logger.info("Deskewed image by %.3f°", angle)
    return deskewed


def enhance_contrast_clahe(
    image: np.ndarray, clip_limit: float = 2.0
) -> np.ndarray:
    """Tingkatkan kontras lokal lewat CLAHE pada channel L (LAB).

    CLAHE diaplikasikan hanya pada channel luminance (L) sehingga keseimbangan
    warna pada channel A & B tetap terjaga — penting untuk struk berwarna
    (mis. logo merchant) maupun screenshot M-Banking yang punya tema warna.

    Args:
        image: Array gambar BGR 3-channel.
        clip_limit: Threshold pembatas kontras untuk CLAHE. Nilai 2.0
            adalah default empiris yang aman untuk teks struk.

    Returns:
        Gambar BGR dengan kontras lokal yang ditingkatkan.

    Raises:
        ValueError: Jika gambar bukan 3-channel.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"CLAHE expects a 3-channel BGR image, got shape {image.shape}"
        )

    # LAB memisahkan luminance (L) dari informasi warna (A, B).
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # tileGridSize=(8, 8) standar untuk gambar dokumen — cukup granular
    # untuk menonjolkan teks redup tanpa over-amplifying noise.
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    logger.debug("Applied CLAHE with clip_limit=%.2f", clip_limit)
    return result


def reduce_noise_gaussian(
    image: np.ndarray, kernel_size: Tuple[int, int] = (3, 3)
) -> np.ndarray:
    """Hilangkan noise frekuensi tinggi dengan Gaussian blur ringan.

    Kernel kecil (default 3×3) dipilih agar tepi karakter tetap terjaga;
    blur agresif justru merusak presisi OCR pada angka & font kecil.

    Args:
        image: Array gambar BGR.
        kernel_size: Ukuran kernel (width, height). Keduanya harus
            bilangan bulat positif ganjil.

    Returns:
        Gambar yang telah di-smoothing.

    Raises:
        ValueError: Jika kernel_size tidak valid (genap atau non-positif).
    """
    kx, ky = kernel_size
    if kx <= 0 or ky <= 0 or kx % 2 == 0 or ky % 2 == 0:
        raise ValueError(
            f"kernel_size dims must be positive odd integers, got {kernel_size}"
        )

    blurred = cv2.GaussianBlur(image, kernel_size, sigmaX=0)
    logger.debug("Gaussian blur applied with kernel %s", kernel_size)
    return blurred


def preprocess_pipeline(
    image_path: str,
    deskew: bool = True,
    enhance: bool = True,
    target_width: int = 1200,
) -> np.ndarray:
    """Pipeline preprocessing end-to-end untuk satu gambar.

    Urutan: load → resize → (deskew) → (CLAHE) → Gaussian denoise.
    Resize dilakukan SEBELUM deskew agar perhitungan sudut konsisten
    pada resolusi target & lebih cepat.

    Args:
        image_path: Path ke file gambar mentah.
        deskew: Jika True, jalankan koreksi kemiringan. Set False untuk
            screenshot M-Banking yang sudah pasti lurus (hemat komputasi).
        enhance: Jika True, jalankan CLAHE. Set False bila gambar sudah
            kontras tinggi atau ingin output mendekati aslinya.
        target_width: Lebar output dalam piksel.

    Returns:
        Array gambar BGR siap diumpankan ke DONUT image processor.

    Raises:
        FileNotFoundError: Jika gambar tidak ditemukan.
        ValueError: Untuk format atau parameter yang tidak valid.
        IOError: Jika gambar gagal di-decode.
    """
    logger.info("Preprocessing started for %s", image_path)

    image = load_image(image_path)
    image = normalize_resolution(image, target_width=target_width)

    if deskew:
        image = deskew_image(image)
    if enhance:
        image = enhance_contrast_clahe(image)

    # Denoising selalu jalan: Gaussian 3×3 cukup ringan untuk semua sumber.
    image = reduce_noise_gaussian(image)

    logger.info(
        "Preprocessing finished for %s (final shape=%s)",
        image_path,
        image.shape,
    )
    return image

@dataclass
class SourceConfig:
    """Konfigurasi preprocessing per sumber data."""
    name: str
    src_dir: Path
    dst_dir: Path
    deskew: bool        # True untuk foto struk, False untuk screenshot
    enhance: bool


def iter_image_files(root: Path) -> list[Path]:
    """Cari semua gambar di bawah `root` secara rekursif (termasuk subfolder)."""
    return [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
    ]


def _process_one(task: tuple) -> tuple[Path, str]:
    """Worker untuk parallel processing. Return (src_path, status)."""
    src, dst, deskew, enhance, target_width = task
    try:
        img = preprocess_pipeline(
            str(src), deskew=deskew, enhance=enhance, target_width=target_width,
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Selalu simpan sebagai PNG (lossless) — input .jpg/.JPG/.webp pun
        # diseragamkan supaya konsumen DONUT konsisten.
        cv2.imwrite(str(dst), img)
        return src, "ok"
    except Exception as exc:                       # noqa: BLE001
        return src, f"failed: {type(exc).__name__}: {exc}"


def process_source(
    cfg: SourceConfig,
    target_width: int = 1200,
    workers: int = 4,
    skip_existing: bool = True,
) -> dict:
    """Proses semua gambar dalam satu sumber data."""
    if not cfg.src_dir.exists():
        logger.error("[%s] source folder tidak ditemukan: %s", cfg.name, cfg.src_dir)
        return {"total": 0, "processed": 0, "skipped": 0, "failed": 0}

    files = iter_image_files(cfg.src_dir)
    if not files:
        logger.warning("[%s] tidak ada gambar di %s", cfg.name, cfg.src_dir)
        return {"total": 0, "processed": 0, "skipped": 0, "failed": 0}

    # Bangun pasangan (src, dst). Yang sudah ada di output akan di-skip.
    tasks = []
    skipped = 0
    for src in files:
        rel = src.relative_to(cfg.src_dir)
        dst = (cfg.dst_dir / rel).with_suffix(".png")
        if skip_existing and dst.exists():
            skipped += 1
            continue
        tasks.append((src, dst, cfg.deskew, cfg.enhance, target_width))

    processed = 0
    failed_log: list[tuple[Path, str]] = []

    if workers <= 1:
        # Mode single-process: lebih mudah debug, jalan di mesin manapun
        for task in tqdm(tasks, desc=cfg.name, unit="img"):
            src, status = _process_one(task)
            if status == "ok":
                processed += 1
            else:
                failed_log.append((src, status))
    else:
        # Mode parallel: gunakan beberapa core CPU
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_one, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=cfg.name, unit="img"):
                src, status = fut.result()
                if status == "ok":
                    processed += 1
                else:
                    failed_log.append((src, status))

    # Tulis daftar file yang gagal — biar bisa diinspeksi manual nanti
    if failed_log:
        cfg.dst_dir.mkdir(parents=True, exist_ok=True)
        log_path = cfg.dst_dir / "_failed.log"
        with open(log_path, "w", encoding="utf-8") as f:
            for src, status in failed_log:
                f.write(f"{src}\t{status}\n")
        logger.info("[%s] daftar gagal → %s", cfg.name, log_path)

    return {
        "total": len(files),
        "processed": processed,
        "skipped": skipped,
        "failed": len(failed_log),
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    PROJECT_ROOT = Path(r"C:\Users\USER\Documents\CAPSTONE")
    OUTPUT_ROOT  = PROJECT_ROOT / "data_processed"      # output utama

    # KONFIGURASI 3 SUMBER DATA -----------------------------------------------
    # Sesuaikan path src_dir kalau struktur foldermu berbeda.
    sources = [
        SourceConfig(
            name="kaggle",
            src_dir=PROJECT_ROOT / "dataset" / "dataset_kaggle"  / "images",
            dst_dir=OUTPUT_ROOT  / "kaggle",
            deskew=True,
            enhance=True,
        ),
        SourceConfig(
            name="cord_hf",
            src_dir=PROJECT_ROOT / "dataset" / "dataset_hf"      / "images",
            dst_dir=OUTPUT_ROOT  / "huggingface",
            deskew=True,
            enhance=True,
        ),
        SourceConfig(
            name="mbanking",
            src_dir=PROJECT_ROOT / "dataset" / "dataset_mbanking" / "images",
            dst_dir=OUTPUT_ROOT  / "mbanking",
            deskew=False,        # screenshot sudah lurus → skip deskew
            enhance=True,
        ),
    ]

    # PARAMETER ---------------------------------------------------------------
    TARGET_WIDTH  = 1200
    WORKERS       = 4            # turunkan ke 1 kalau ingin debug step-by-step
    SKIP_EXISTING = True         # set False untuk re-process semua

    # JALANKAN PER SUMBER -----------------------------------------------------
    summary = {}
    for cfg in sources:
        logger.info("=" * 72)
        logger.info("[%s] %s  →  %s", cfg.name, cfg.src_dir, cfg.dst_dir)
        summary[cfg.name] = process_source(
            cfg,
            target_width=TARGET_WIDTH,
            workers=WORKERS,
            skip_existing=SKIP_EXISTING,
        )

    # RINGKASAN AKHIR ---------------------------------------------------------
    logger.info("=" * 72)
    logger.info("RINGKASAN BATCH PREPROCESSING")
    logger.info("%-14s %8s %10s %8s %8s",
                "source", "total", "processed", "skipped", "failed")
    for name, s in summary.items():
        logger.info("%-14s %8d %10d %8d %8d",
                    name, s["total"], s["processed"], s["skipped"], s["failed"])