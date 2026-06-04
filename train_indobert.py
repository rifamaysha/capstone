"""Fine-tune IndoBERT untuk klasifikasi kategori transaksi (5 kelas).

Pakai class_weight untuk handle imbalance (makanan_minuman dominan ~55%).
Output: model checkpoint + classification report + confusion matrix.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    Trainer, TrainingArguments,
)

from indobert import (
    ACTIVE_CLASSES, DATA_DIR, MODEL_OUT, TrainingConfig, CategoryDataset,
)

logger = logging.getLogger(__name__)


class WeightedTrainer(Trainer):
    """Trainer subclass — pakai class_weight di CrossEntropy loss."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = nn.CrossEntropyLoss(weight=weight)(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred) -> dict[str, float]:
    """Metrics dipanggil Trainer setiap epoch eval."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":    accuracy_score(labels, preds),
        "f1_macro":    f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    cfg = TrainingConfig()
    label_to_id = {l: i for i, l in enumerate(ACTIVE_CLASSES)}
    id_to_label = {i: l for l, i in label_to_id.items()}

    logger.info("Active classes: %s", ACTIVE_CLASSES)
    logger.info("Loading model: %s", cfg.base_model)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model,
        num_labels=len(ACTIVE_CLASSES),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    # ---- Datasets ----
    train_ds = CategoryDataset(DATA_DIR / "train_labeled.jsonl",
                                tokenizer, label_to_id, cfg.max_length)
    val_ds   = CategoryDataset(DATA_DIR / "val_labeled.jsonl",
                                tokenizer, label_to_id, cfg.max_length)
    test_ds  = CategoryDataset(DATA_DIR / "test_labeled.jsonl",
                                tokenizer, label_to_id, cfg.max_length)
    logger.info("Train=%d  Val=%d  Test=%d",
                len(train_ds), len(val_ds), len(test_ds))

    # ---- Distribusi & class weights ----
    train_labels = [s["label"] for s in train_ds.samples]
    cnt = Counter(train_labels)
    logger.info("Train distribution:")
    for label in ACTIVE_CLASSES:
        n = cnt.get(label, 0)
        logger.info("  %-18s %4d (%.1f%%)", label, n, 100*n/len(train_ds))

    class_weights = None
    if cfg.use_class_weight:
        ids = np.array([label_to_id[l] for l in train_labels])
        classes_arr = np.arange(len(ACTIVE_CLASSES))
        weights = compute_class_weight(
            "balanced", classes=classes_arr, y=ids,
        )
        class_weights = torch.tensor(weights, dtype=torch.float32)
        logger.info("Class weights: %s",
                    [f"{w:.2f}" for w in class_weights.tolist()])

    # ---- Trainer setup ----
    output_dir = MODEL_OUT / "run1"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=cfg.save_total_limit,
        logging_steps=20,
        seed=cfg.seed,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    logger.info("=" * 72)
    logger.info("STARTING TRAINING")
    logger.info("=" * 72)
    trainer.train()

    # ---- Eval di test set ----
    logger.info("=" * 72)
    logger.info("EVALUATING ON TEST SET")
    logger.info("=" * 72)
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    for k, v in test_metrics.items():
        logger.info("  %s: %.4f", k, v)

    test_preds = trainer.predict(test_ds)
    pred_ids = np.argmax(test_preds.predictions, axis=-1)
    true_ids = test_preds.label_ids
    target_names = [id_to_label[i] for i in range(len(ACTIVE_CLASSES))]

    logger.info("\nClassification report (test):\n%s",
                classification_report(true_ids, pred_ids,
                                       target_names=target_names,
                                       zero_division=0, digits=3))

    cm = confusion_matrix(true_ids, pred_ids,
                          labels=list(range(len(ACTIVE_CLASSES))))
    logger.info("Confusion matrix (rows=true, cols=pred):")
    header = " " * 14 + "  ".join(f"{n[:8]:>8s}" for n in target_names)
    logger.info(header)
    for i, row in enumerate(cm):
        cells = "  ".join(f"{x:>8d}" for x in row)
        logger.info("%-14s%s", target_names[i][:14], cells)

    # ---- Save final ----
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    metadata = {
        "active_classes": ACTIVE_CLASSES,
        "label_to_id":    label_to_id,
        "id_to_label":    {str(k): v for k, v in id_to_label.items()},
        "test_metrics":   {k: float(v) for k, v in test_metrics.items()},
    }
    with open(final_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Saved → %s", final_dir)


if __name__ == "__main__":
    main()