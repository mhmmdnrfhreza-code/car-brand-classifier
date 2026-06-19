from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from config import CLASS_INDICES_PATH, MODEL_PATH, TOP_K
from dataset import load_class_indices
from preprocessing import make_batch, preprocess_bytes


class ModelNotReadyError(Exception):
    """Model atau class_indices belum dimuat / belum tersedia."""


_model = None
_index_to_label: Optional[Dict[int, str]] = None


def load_artifacts(model_path=MODEL_PATH, class_indices_path=CLASS_INDICES_PATH) -> bool:
    """Muat model Keras + class_indices ke memori. Return True bila berhasil."""
    global _model, _index_to_label
    model_path = Path(model_path)
    class_indices_path = Path(class_indices_path)
    if not model_path.exists() or not class_indices_path.exists():
        return False
    import tensorflow as tf  # lazy import

    _model = tf.keras.models.load_model(model_path)
    class_indices = load_class_indices(class_indices_path)
    _index_to_label = {int(idx): label for label, idx in class_indices.items()}
    return True


def is_ready() -> bool:
    return _model is not None and _index_to_label is not None


def predict(image_bytes: bytes, top_k: int = TOP_K, model=None, index_to_label=None):
    """Prediksi brand dari bytes gambar. Mengembalikan dict label/confidence/top_k.

    `model` & `index_to_label` dapat diinjeksi (untuk pengujian).
    """
    model = model if model is not None else _model
    index_to_label = index_to_label if index_to_label is not None else _index_to_label
    if model is None or index_to_label is None:
        raise ModelNotReadyError("Model belum dimuat. Latih model lebih dulu.")

    arr = preprocess_bytes(image_bytes)  # raise InvalidImageError bila tidak valid
    batch = make_batch(arr)
    preds = np.asarray(model.predict(batch, verbose=0))[0].astype("float32")

    k = max(1, min(int(top_k), len(index_to_label)))
    top_idx = np.argsort(preds)[::-1][:k]
    top = [
        {"label": index_to_label[int(i)], "confidence": float(preds[int(i)])}
        for i in top_idx
    ]
    return {
        "label": top[0]["label"],
        "confidence": top[0]["confidence"],
        "top_k": top,
    }
