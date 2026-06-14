"""Safe HuggingFace category classifier adapter.

Loads a BertForSequenceClassification checkpoint from models/indobert/run1/final
if the weights are valid. Falls back gracefully when:
  - the model directory is missing
  - the safetensors weights are corrupted/placeholder
  - transformers / torch are not importable
  - any runtime error occurs during inference

No brand-specific keywords. No dependency on the legacy indobert/hybrid.py
chain (which has missing source files auto_labeler.py / inference.py).
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Result schema used by callers
UNAVAILABLE_RESULT: dict[str, Any] = {
    "label": "lainnya",
    "confidence": 0.0,
    "source": "ml_unavailable",
}


class SafeCategoryClassifier:
    """Singleton wrapper around a HuggingFace sequence-classification model."""

    _instance: "SafeCategoryClassifier | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self._loaded = False
        self._load_error: str = ""
        self._tokenizer: Any = None
        self._model: Any = None
        self._id2label: dict[int, str] = {}
        self._torch: Any = None

    @classmethod
    def get(cls, model_dir: Path) -> "SafeCategoryClassifier":
        """Return process-wide singleton, building it on first call."""
        if cls._instance is not None:
            return cls._instance
        with cls._instance_lock:
            if cls._instance is None:
                inst = cls(model_dir)
                inst._try_load()
                cls._instance = inst
        return cls._instance

    def _try_load(self) -> None:
        """Attempt to load tokenizer + weights. Never raises."""
        try:
            if not self.model_dir.exists():
                self._load_error = f"model_dir_missing: {self.model_dir}"
                logger.info("SafeCategoryClassifier: %s", self._load_error)
                return

            weights_path = self.model_dir / "model.safetensors"
            if weights_path.exists() and weights_path.stat().st_size < 1_000_000:
                # A real BERT classifier weight file is hundreds of MB.
                # A <1 MB file is a placeholder / corrupted checkpoint — skip.
                self._load_error = (
                    f"weights_too_small: {weights_path.stat().st_size} bytes "
                    "(expected ~100+ MB)"
                )
                logger.info("SafeCategoryClassifier: %s", self._load_error)
                return
        except (OSError, PermissionError) as exc:
            self._load_error = f"filesystem_error: {type(exc).__name__}: {exc}"
            logger.warning("SafeCategoryClassifier: %s", self._load_error)
            return

        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except Exception as exc:
            self._load_error = f"dependency_unavailable: {type(exc).__name__}: {exc}"
            logger.warning("SafeCategoryClassifier: %s", self._load_error)
            return

        try:
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
            model.eval()
        except Exception as exc:
            self._load_error = f"model_load_failed: {type(exc).__name__}: {exc}"
            logger.warning("SafeCategoryClassifier: %s", self._load_error)
            return

        id2label_raw = getattr(model.config, "id2label", {}) or {}
        try:
            id2label = {int(k): str(v) for k, v in id2label_raw.items()}
        except Exception:
            self._load_error = "invalid_id2label"
            logger.warning("SafeCategoryClassifier: %s", self._load_error)
            return

        if not id2label:
            self._load_error = "empty_id2label"
            logger.warning("SafeCategoryClassifier: %s", self._load_error)
            return

        self._tokenizer = tokenizer
        self._model = model
        self._id2label = id2label
        self._torch = torch
        self._loaded = True
        logger.info(
            "SafeCategoryClassifier loaded labels=%s", list(id2label.values())
        )

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> str:
        return self._load_error

    def predict(self, text: str, *, min_confidence: float = 0.55) -> dict[str, Any]:
        """Classify ``text``.

        Returns a dict ``{label, confidence, source}``. When the model is not
        available or confidence is below ``min_confidence``, returns
        ``UNAVAILABLE_RESULT`` so callers can safely fall back to ``lainnya``.

        The ``source`` field tells the caller which path produced the answer:
          - ``"ml_classifier"``        — model output above threshold
          - ``"ml_low_confidence"``    — model output below threshold
          - ``"ml_unavailable"``       — model could not be loaded
          - ``"empty_input"``          — nothing to classify
        """
        if not self._loaded:
            return dict(UNAVAILABLE_RESULT)

        cleaned = (text or "").strip()
        if not cleaned:
            return {"label": "lainnya", "confidence": 0.0, "source": "empty_input"}

        try:
            torch = self._torch
            with torch.no_grad():
                enc = self._tokenizer(
                    cleaned,
                    return_tensors="pt",
                    truncation=True,
                    max_length=64,
                )
                logits = self._model(**enc).logits[0]
                probs = torch.softmax(logits, dim=-1)
                idx = int(torch.argmax(probs))
                conf = float(probs[idx])
        except Exception as exc:
            logger.warning("SafeCategoryClassifier.predict failed: %s", exc)
            return dict(UNAVAILABLE_RESULT)

        label = self._id2label.get(idx, "lainnya")
        if conf < min_confidence:
            return {
                "label": "lainnya",
                "confidence": conf,
                "source": "ml_low_confidence",
            }
        return {"label": label, "confidence": conf, "source": "ml_classifier"}
