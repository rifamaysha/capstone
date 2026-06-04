"""Konfigurasi training DONUT."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\USER\Documents\CAPSTONE")
DATA_DIR     = PROJECT_ROOT / "data_processed" / "unified"
MODEL_OUT    = PROJECT_ROOT / "models" / "donut"


@dataclass
class TrainingConfig:
    """Hyperparameter & path training DONUT.

    Default sudah disesuaikan untuk dataset ~1000 sample dan GPU 8-12 GB.
    """
    # Base model — pakai yang sudah pretrained di CORD karena 800/1029 data kita CORD-style
    base_model: str = "naver-clova-ix/donut-base-finetuned-cord-v2"

    # Image — DONUT image size (h, w). Resize otomatis oleh processor.
    image_height: int = 1280
    image_width:  int = 960

    # Sequence — max length untuk target string (XML tags + items bisa panjang)
    max_target_length: int = 768

    # Optimization
    num_epochs: int           = 10
    batch_size: int           = 2          # turunkan ke 1 kalau OOM
    grad_accum_steps: int     = 4          # effective batch = 2 * 4 = 8
    learning_rate: float      = 3e-5
    warmup_ratio: float       = 0.05
    weight_decay: float       = 0.01

    # Hardware
    fp16: bool        = True               # mixed precision → hemat VRAM
    num_workers: int  = 0                  # Windows: 0 lebih aman daripada >0

    # Bookkeeping
    save_total_limit: int  = 2
    seed: int              = 42