from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import config
import predictor
from preprocessing import InvalidImageError


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        loaded = predictor.load_artifacts()
        if not loaded:
            print("[warn] Model belum ada. Jalankan train.py lalu restart server.")
    except Exception as exc:  # noqa: BLE001 - jangan gagalkan startup
        print(f"[warn] Gagal memuat model: {exc}")
    yield


app = FastAPI(title="Car Brand Logo Classifier", version="1.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": predictor.is_ready()}


@app.get("/classes")
def classes():
    """Daftar kelas/brand yang dikenali model (urut indeks)."""
    if not predictor.is_ready() or not predictor._index_to_label:
        return {"classes": []}
    labels = [
        predictor._index_to_label[i]
        for i in sorted(predictor._index_to_label.keys())
    ]
    return {"classes": labels}


@app.get("/metrics")
def metrics():
    """Hasil evaluasi terakhir (akurasi + confusion matrix)."""
    path = config.METRICS_PATH
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Metrics belum tersedia. Jalankan train.py atau evaluate.py.",
        )
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Ekstensi tidak didukung (JPG/PNG).")
    if file.content_type not in config.ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Tipe konten tidak didukung.")

    data = await file.read()
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"Ukuran melebihi {config.MAX_UPLOAD_MB} MB."
        )

    try:
        result = predictor.predict(data)
    except predictor.ModelNotReadyError:
        raise HTTPException(
            status_code=503,
            detail="Model belum tersedia. Latih model dulu dengan train.py.",
        )
    except InvalidImageError:
        raise HTTPException(status_code=400, detail="File bukan gambar valid.")
    return result


# Layani frontend statis (paling akhir agar tidak menimpa route API di atas).
if config.FRONTEND_DIR.exists():
    app.mount(
        "/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend"
    )
