"""PyTorch Dataset untuk fine-tuning DONUT pada unified receipts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import DonutProcessor


class DonutReceiptDataset(Dataset):
    """Wrap unified JSONL → sample siap masuk DONUT.

    Setiap baris JSONL berisi `image_path` (path absolut ke PNG processed)
    dan `donut_target` (string XML-tagged hasil encode_donut_target).

    __getitem__ returns dict:
        pixel_values: Tensor[3, H, W]   — gambar setelah image processor
        labels:       Tensor[seq_len]    — token id target, padding = -100
    """

    def __init__(
        self,
        jsonl_path: Path,
        processor: DonutProcessor,
        max_target_length: int = 768,
    ) -> None:
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

        self.processor = processor
        self.max_target_length = max_target_length

        # In-memory: 1029 record × ~1 KB metadata = aman
        self.entries: list[dict[str, Any]] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.entries.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        entry = self.entries[idx]

        # Load gambar processed (sudah deskewed + CLAHE dari Bagian 1)
        image = Image.open(entry["image_path"]).convert("RGB")
        pixel_values = self.processor(
            image, return_tensors="pt",
        ).pixel_values.squeeze(0)

        # Target = donut_target + EOS. Decoder akan generate ini autoregresif.
        target = entry["donut_target"] + self.processor.tokenizer.eos_token

        token_ids = self.processor.tokenizer(
            target,
            add_special_tokens=False,        # tidak auto-tambah BOS/EOS, kita kontrol manual
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Mask padding agar tidak dihitung di loss
        labels = token_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}