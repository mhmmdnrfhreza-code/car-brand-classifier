from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

# Folder dataset. Setelah mengekstrak dataset Kaggle, letakkan isinya di
# <project_root>/dataset (loader otomatis menelusuri folder pembungkus & Train/
# Test). Override dengan env IMAGES_DIR bila lokasinya berbeda.
IMAGES_DIR = Path(os.environ.get("IMAGES_DIR", str(PROJECT_ROOT / "dataset")))

MODELS_DIR = BACKEND_DIR / "models"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(MODELS_DIR / "model.keras")))
CLASS_INDICES_PATH = Path(
    os.environ.get("CLASS_INDICES_PATH", str(MODELS_DIR / "class_indices.json"))
)
METRICS_PATH = MODELS_DIR / "metrics.json"  # hasil evaluasi (akurasi + confusion matrix)

FRONTEND_DIR = PROJECT_ROOT / "frontend"

# --- Image / preprocessing ---------------------------------------------------
IMG_SIZE = (224, 224)  # (height, width) input standar MobileNetV2
CHANNELS = 3

# --- Training hyperparameters ------------------------------------------------
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
EPOCHS = int(os.environ.get("EPOCHS", "20"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
DROPOUT = 0.2
LABEL_SMOOTHING = 0.1
VAL_SPLIT = 0.15  # porsi validasi yang dipisahkan dari data Train
TEST_SPLIT = 0.15  # hanya dipakai bila dataset TIDAK punya folder Test sendiri
SEED = 42

# Hanya latih kelas dengan >= N gambar (1 = semua kelas). Dataset ini seimbang,
# jadi default 1 sudah aman.
MIN_IMAGES_PER_CLASS = int(os.environ.get("MIN_IMAGES_PER_CLASS", "1"))

# Fine-tuning opsional: jumlah layer teratas base model yang di-unfreeze.
# 0 = tanpa fine-tuning (hanya melatih classification head).
# Direkomendasikan ~30 untuk akurasi lebih tinggi.
FINE_TUNE_AT = int(os.environ.get("FINE_TUNE_AT", "0"))
FINE_TUNE_EPOCHS = int(os.environ.get("FINE_TUNE_EPOCHS", "10"))

# --- Inference / API ---------------------------------------------------------
TOP_K = 3
MAX_UPLOAD_MB = 10
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# CORS: origin yang diizinkan (pisahkan dengan koma pada env CORS_ORIGINS).
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000,"
        "http://localhost:5500,http://127.0.0.1:5500",
    ).split(",")
    if o.strip()
]
