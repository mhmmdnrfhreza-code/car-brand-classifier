# Brand Classifier

Meskipun Kerangka Kerja Klasifikasi Logo Brand Berbasis **MobileNetV2** + **FastAPI** Saat Ini Dilatih Untuk Logo Mobil, Namun Dapat Digunakan Juga Untuk Kasus Lain Yang Lebih Luas. Seperti, Mengenali Logo Brand Populer Dari Berbagai Bidang/Sektor Global (F&B, Finance, Fashion, dan Sebagainya) Cukup Dengan Menyesuaikan Datasetnya.

Dataset: [Car Brand Logos](https://www.kaggle.com/datasets/volkandl/car-brand-logos) — 8 kelas: `Hyundai, Lexus, Mazda, Mercedes, Opel, Skoda, Toyota, Volkswagen`.

## Fitur

- Transfer Learning MobileNetV2 + Fine-Tuning Opsional.
- Deteksi Struktur Dataset Otomatis (`Train/Test` atau Folder Per Kelas).
- REST API: Prediksi Top-3 dan Metrik Evaluasi.
- Web Interface: Upload Gambar, Akurasi + Confusion Matrix.
- Normalisasi Piksel Konsisten (`-1..1`) Antara Rraining dan Inferensi.

## Hasil

| Tahap         | Akurasi | Loss   |
| ------------- | ------- | ------ |
| Head Only     | 0.7100  | 1.2015 |
| + Fine-Tuning | 0.8175  | 1.0545 |

## Struktur

```
car-brand-classifier/
├─ backend/
│  ├─ config.py          # Konstanta & Hyperparameter
│  ├─ preprocessing.py   # Preprocessing Gambar (-1..1)
│  ├─ dataset.py         # Scan Folder, Split, tf.data Pipeline
│  ├─ metrics.py         # Confusion Matrix & Precision/Recall/F1
│  ├─ train.py           # Training + Fine-Tuning + Tulis metrics.json
│  ├─ evaluate.py        # Hitung Ulang Metrik Dari Model Tersimpan
│  ├─ predictor.py       # Inferensi Top-K
│  ├─ main.py            # FastAPI: /health /classes /metrics /predict
│  ├─ requirements.txt
│  ├─ models/            # Artefak Hasil Training (model.keras, *.json)
│  └─ tests/             # Unit Test (Tanpa TensorFlow)
├─ frontend/             # index.html, css/, js/
└─ dataset/              # Diisi sendiri (Lihat "Dataset")
```

## Instalasi

```bash
cd backend
python -m venv .venv
# Linux/Mac: source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Dataset Tidak Disertakan Dalam Repo. Download Lalu Ekstrak Ke `dataset/`:

```bash
pip install kaggle
kaggle datasets download -d volkandl/car-brand-logos -p dataset --unzip
```
dataset/
└─ Car_Brand_Logos/
   ├─ Train/<brand>/*.jpg
   └─ Test/<brand>/*.jpg
```

Jika Hanya Tersedia Subfolder Per Kelas Tanpa `Train/Test`, Data Dibagi Otomatis Secara Stratified. Lokasi Dataset Dapat Ditimpa Lewat Variabel Lingkungan `IMAGES_DIR`.

## Penggunaan

**Training**

```bash
cd backend
python train.py --fine-tune-at 30 --fine-tune-epochs 10
```

Menghasilkan `models/model.keras`, `models/class_indices.json`, dan `models/metrics.json`.

Argumen Utama: `--epochs`, `--batch-size`, `--learning-rate`, `--min-images-per-class`, `--fine-tune-at`, `--fine-tune-epochs`, `--val-split`.

**Evaluasi Ulang** (Tanpa Melatih Ulang, Memperbarui `metrics.json`):

```bash
python evaluate.py
```

**Menjalankan Server**

```bash
uvicorn main:app --reload --port 8000
```

Buka <http://127.0.0.1:8000>

## API

| Method | Endpoint   | Keterangan                                   |
| ------ | ---------- | -------------------------------------------- |
| GET    | `/health`  | Status API & kesiapan model                  |
| GET    | `/classes` | Daftar kelas/brand yang dikenali             |
| GET    | `/metrics` | Akurasi, loss, confusion matrix, per-kelas   |
| POST   | `/predict` | Multipart `file` → prediksi Top-3            |

Contoh Respons `/predict`:

```json
{
  "label": "toyota",
  "confidence": 0.93,
  "top_k": [
    { "label": "toyota", "confidence": 0.93 },
    { "label": "lexus", "confidence": 0.04 },
    { "label": "mazda", "confidence": 0.02 }
  ]
}
```

## Pengujian

```bash
cd backend
python -m unittest discover -s tests -t .
```

Mencakup Preprocessing, Pemuatan Dataset, Metrik Evaluasi, dan Logika Predictor - Tanpa Memerlukan TensorFlow Maupun Dataset Asli.

## Catatan

- Training Dan Inferensi Memakai Normalisasi `0..255 → -1..1` (Konvensi MobileNetV2). Bila Pernah Melatih Dengan Skema Lama (`/255`), Hapus `models/Model.keras` Dan `models/Class_indices.json` Sebelum Melatih Ulang.
- Pesan "GPU Support Is Not Available On Native Windows" Bersifat Informatif, Training Berjalan Di CPU. Untuk GPU Gunakan WSL2.

## Lisensi

MIT.

## Tim

| <img src="https://github.com/rizqidimas.png" width="80"><br>[Dimas Rizqi Purnomo](https://github.com/rizqidimas) |