from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from config import (
    BATCH_SIZE,
    CLASS_INDICES_PATH,
    IMAGES_DIR,
    IMG_SIZE,
    MIN_IMAGES_PER_CLASS,
    SEED,
    TEST_SPLIT,
    VAL_SPLIT,
)

# (absolute_image_path, label)
Sample = Tuple[str, str]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_NAMES = {"train", "training"}
TEST_NAMES = {"test", "testing", "val", "valid", "validation"}


# --- Penelusuran folder ------------------------------------------------------
def _subdirs(path) -> List[Path]:
    path = Path(path)
    if not path.exists():
        return []
    return sorted([d for d in path.iterdir() if d.is_dir()])


def _has_images(directory) -> bool:
    return any(
        f.is_file() and f.suffix.lower() in IMAGE_EXTS for f in Path(directory).iterdir()
    )


def _has_class_subdirs(path) -> bool:
    return any(_has_images(d) for d in _subdirs(path))


def _find_named(path, names) -> Path | None:
    for d in _subdirs(path):
        if d.name.lower() in names:
            return d
    return None


def resolve_images_dir(images_dir=IMAGES_DIR) -> Path:
    """Telusuri folder pembungkus tunggal hingga menemukan struktur dataset.

    Berhenti saat menemukan folder yang memuat Train/Test ATAU subfolder kelas.
    """
    root = Path(images_dir)
    for _ in range(4):
        if not root.exists():
            break
        if _find_named(root, TRAIN_NAMES) or _find_named(root, TEST_NAMES):
            return root
        if _has_class_subdirs(root):
            return root
        subs = _subdirs(root)
        if len(subs) == 1:
            root = subs[0]
            continue
        break
    return Path(images_dir)


def load_samples_from_dir(directory) -> List[Sample]:
    """Pindai subfolder kelas -> daftar (absolute_image_path, label)."""
    samples: List[Sample] = []
    for class_dir in _subdirs(directory):
        label = class_dir.name.strip()
        if not label:
            continue
        for f in sorted(class_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                samples.append((str(f.resolve()), label))
    return samples


def discover_dataset(images_dir=IMAGES_DIR):
    """Kembalikan (mode, train_dir, test_dir, root).

    mode == "split" bila ada folder Train (dan mungkin Test);
    mode == "flat" bila hanya ada subfolder kelas.
    """
    root = resolve_images_dir(images_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan: {root}. "
            "Ekstrak dataset ke <project_root>/dataset atau set env IMAGES_DIR."
        )
    train_dir = _find_named(root, TRAIN_NAMES)
    test_dir = _find_named(root, TEST_NAMES)
    if train_dir is not None:
        return "split", train_dir, test_dir, root
    if _has_class_subdirs(root):
        return "flat", None, None, root
    raise FileNotFoundError(
        f"Tidak ada subfolder kelas berisi gambar di: {root}."
    )


# --- Transformasi sampel -----------------------------------------------------
def filter_min_images(samples: List[Sample], min_count: int) -> List[Sample]:
    """Sisakan hanya sampel dari kelas dengan >= min_count gambar (1 = tanpa filter)."""
    if min_count is None or min_count <= 1:
        return list(samples)
    counts = Counter(label for _, label in samples)
    return [s for s in samples if counts[s[1]] >= min_count]


def class_distribution(samples: List[Sample]) -> Dict[str, int]:
    return dict(Counter(label for _, label in samples))


def build_label_index(samples: List[Sample]) -> Dict[str, int]:
    """Mapping label -> id (urut alfabet, tanpa hardcode)."""
    labels = sorted({label for _, label in samples})
    return {label: idx for idx, label in enumerate(labels)}


def _group_by_label(samples: List[Sample]) -> Dict[str, List[Sample]]:
    groups: Dict[str, List[Sample]] = {}
    for s in samples:
        groups.setdefault(s[1], []).append(s)
    return groups


def split_train_val(samples, val_split=VAL_SPLIT, seed=SEED):
    """Pisahkan validasi dari data train, per-label (stratified)."""
    rng = random.Random(seed)
    train: List[Sample] = []
    val: List[Sample] = []
    for label, items in sorted(_group_by_label(samples).items()):
        items = items[:]
        rng.shuffle(items)
        n = len(items)
        n_val = int(round(n * val_split))
        if n >= 2:
            n_val = max(1, min(n_val, n - 1))
        else:
            n_val = 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


def stratified_split(samples, val_split=VAL_SPLIT, test_split=TEST_SPLIT, seed=SEED):
    """Bagi menjadi train/val/test per-label agar tiap kelas terwakili."""
    rng = random.Random(seed)
    train: List[Sample] = []
    val: List[Sample] = []
    test: List[Sample] = []
    for label, items in sorted(_group_by_label(samples).items()):
        items = items[:]
        rng.shuffle(items)
        n = len(items)
        n_test = int(round(n * test_split))
        n_val = int(round(n * val_split))
        if n >= 3:
            n_test = max(1, n_test)
            n_val = max(1, n_val)
            while n_test + n_val >= n:
                if n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1
                else:
                    break
        elif n == 2:
            n_test, n_val = 1, 0
        else:
            n_test, n_val = 0, 0
        test.extend(items[:n_test])
        val.extend(items[n_test : n_test + n_val])
        train.extend(items[n_test + n_val :])
    return train, val, test


def prepare_splits(images_dir=IMAGES_DIR, min_images=MIN_IMAGES_PER_CLASS, val_split=VAL_SPLIT):
    """Kembalikan (train, val, test, mode) sesuai struktur dataset.

    - mode "split": validasi dipisah dari Train, Test dipakai apa adanya.
    - mode "flat": dibagi otomatis train/val/test secara stratified.
    """
    mode, train_dir, test_dir, root = discover_dataset(images_dir)
    if mode == "split":
        train_all = filter_min_images(load_samples_from_dir(train_dir), min_images)
        test_samples = load_samples_from_dir(test_dir) if test_dir else []
        train_samples, val_samples = split_train_val(train_all, val_split)
    else:
        all_samples = filter_min_images(load_samples_from_dir(root), min_images)
        train_samples, val_samples, test_samples = stratified_split(all_samples)
    return train_samples, val_samples, test_samples, mode


# --- Persistensi class_indices ----------------------------------------------
def save_class_indices(class_indices: Dict[str, int], path=CLASS_INDICES_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(class_indices, f, ensure_ascii=False, indent=2)


def load_class_indices(path=CLASS_INDICES_PATH) -> Dict[str, int]:
    with open(Path(path), encoding="utf-8") as f:
        return json.load(f)


# --- tf.data (TensorFlow di-import lazy) -------------------------------------
def _build_augmenter():
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ],
        name="augmenter",
    )


def make_tf_dataset(
    samples,
    class_indices,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    augment=False,
):
    """Bangun tf.data.Dataset dari daftar sampel.

    Normalisasi identik dengan preprocessing.py: piksel /127.5 - 1.0 (-1..1).
    """
    import tensorflow as tf

    paths = [p for p, _ in samples]
    labels = [class_indices[l] for _, l in samples]
    num_classes = len(class_indices)
    augmenter = _build_augmenter() if augment else None

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=max(1, len(samples)), seed=SEED, reshuffle_each_iteration=True
        )

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 127.5 - 1.0
        if augmenter is not None:
            img = augmenter(img)
        return img, tf.one_hot(label, num_classes)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
