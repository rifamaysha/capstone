"""Inference DONUT pretrained pada gambar struk.

Pakai model 'donut-base-finetuned-cord-v2' yang sudah dilatih di CORD,
cocok untuk struk Indonesia. Tidak ada training; langsung pakai.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

logger = logging.getLogger(__name__)


class ReceiptParser:
    """Wrapper minimal untuk inference DONUT pretrained.

    Single instance bisa dipakai untuk multiple gambar — model di-load
    sekali di __init__ (slow, ~700 MB download di run pertama), lalu
    method parse() bisa dipanggil berkali-kali (cepat per gambar).
    """

    DEFAULT_MODEL = "naver-clova-ix/donut-base-finetuned-cord-v2"
    TASK_PROMPT   = "<s_cord-v2>"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
    ) -> None:
        """
        Args:
            model_name: HuggingFace model id atau path lokal ke checkpoint.
            device: 'cuda' / 'cpu' / None (auto-detect).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading DONUT (%s) on %s", model_name, self.device)

        self.processor = DonutProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def parse(self, image_path: str | Path) -> dict[str, Any]:
        """Parse satu gambar struk → dict terstruktur.

        Args:
            image_path: Path ke gambar (raw atau processed dari Bagian 1).

        Returns:
            Dict hasil parse, contoh untuk struk:
                {
                  "menu": [{"nm": "...", "cnt": "1", "price": "10000"}, ...],
                  "total": {"total_price": "..."}
                }

        Raises:
            FileNotFoundError: gambar tidak ada di path.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        # Encoder input: pixel values
        pixel_values = self.processor(
            image, return_tensors="pt",
        ).pixel_values.to(self.device)

        # Decoder priming: task prompt agar model tahu format yg diharapkan
        decoder_input_ids = self.processor.tokenizer(
            self.TASK_PROMPT,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.model.decoder.config.max_position_embeddings,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                # Cegah model output token <unk> yang akan bikin parse gagal
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        # Decode token → string XML-tagged
        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "")
        sequence = sequence.replace(self.processor.tokenizer.pad_token, "")
        # Buang task prompt token di awal (cuma yg pertama, count=1)
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

        # Convert XML-tagged string → nested dict (built-in helper DONUT)
        return self.processor.token2json(sequence)