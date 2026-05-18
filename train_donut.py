"""Fine-tune DONUT pada unified receipt dataset.

Jalankan:
    python train_donut.py            # full training
    python train_donut.py --smoke    # quick test 1 epoch, subset 10 sampel
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset
from transformers import (
    DonutProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
)

from donut_finetune import DATA_DIR, MODEL_OUT, DonutReceiptDataset, TrainingConfig
from loaders import collect_special_tokens

logger = logging.getLogger(__name__)


def setup_model_and_processor(cfg: TrainingConfig):
    """Load base DONUT, register special tokens, resize embeddings."""
    logger.info("Loading base model: %s", cfg.base_model)
    processor = DonutProcessor.from_pretrained(cfg.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(cfg.base_model)

    # Set ukuran gambar yang akan dipakai image processor.
    # do_align_long_axis=False: gambar kita sudah portrait/landscape "alami".
    processor.image_processor.size = {
        "height": cfg.image_height,
        "width":  cfg.image_width,
    }
    processor.image_processor.do_align_long_axis = False

    # Daftarkan special token kustom kita ke tokenizer (<s_total>, <s_item>, dll.)
    new_tokens = collect_special_tokens()
    n_added = processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": new_tokens}
    )
    logger.info("Registered %d custom special tokens", n_added)

    # Resize embedding decoder agar token baru punya vector
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # Konfigurasi token spesial untuk generation/training
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    # Decoder start = BOS standar dari tokenizer
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s>")

    return processor, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Quick test: 1 epoch, 10 train sampel, 4 val sampel")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    # ---- Sanity check GPU ----
    if torch.cuda.is_available():
        logger.info("GPU: %s (%.1f GB)",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)
    else:
        logger.warning("CUDA tidak tersedia → training akan SANGAT lambat di CPU")

    cfg = TrainingConfig()
    if args.smoke:
        cfg.num_epochs = 1
        logger.info("=== SMOKE TEST MODE ===")

    processor, model = setup_model_and_processor(cfg)

    # ---- Datasets ----
    train_ds = DonutReceiptDataset(DATA_DIR / "train.jsonl", processor, cfg.max_target_length)
    val_ds   = DonutReceiptDataset(DATA_DIR / "val.jsonl",   processor, cfg.max_target_length)

    if args.smoke:
        train_ds = Subset(train_ds, range(min(10, len(train_ds))))
        val_ds   = Subset(val_ds,   range(min(4,  len(val_ds))))

    logger.info("Train: %d | Val: %d", len(train_ds), len(val_ds))

    # ---- Training arguments ----
    output_dir = models / ("smoke" if args.smoke else "run1")
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=cfg.save_total_limit,
        logging_steps=10,
        seed=cfg.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=cfg.num_workers,
        predict_with_generate=False,        # eval pakai loss saja (lebih cepat)
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    logger.info("Starting training…")
    trainer.train()

    # ---- Save final ----
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    logger.info("Model + processor saved → %s", final_dir)


if __name__ == "__main__":
    main()