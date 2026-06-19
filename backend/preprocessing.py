from __future__ import annotations

import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from config import IMG_SIZE


class InvalidImageError(Exception):
    """File yang diunggah bukan gambar valid / tidak dapat dibaca."""


def preprocess_pil(img: "Image.Image", img_size=IMG_SIZE) -> np.ndarray:
    """Konversi PIL Image -> array float32 (H, W, 3) ternormalisasi ke -1..1."""
    # PIL.resize menerima (width, height); IMG_SIZE adalah (height, width).
    img = img.convert("RGB").resize((img_size[1], img_size[0]))
    arr = np.asarray(img, dtype="float32")
    arr = arr / 127.5 - 1.0  # setara mobilenet_v2.preprocess_input
    return arr


def preprocess_bytes(data: bytes, img_size=IMG_SIZE) -> np.ndarray:
    """Decode bytes gambar lalu preprocess. Raise InvalidImageError bila gagal."""
    if not data:
        raise InvalidImageError("File kosong.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("File bukan gambar valid atau rusak.") from exc
    return preprocess_pil(img, img_size)


def make_batch(arr: np.ndarray) -> np.ndarray:
    """Tambah dimensi batch: (H, W, 3) -> (1, H, W, 3)."""
    return np.expand_dims(arr, axis=0)
