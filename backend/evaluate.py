from __future__ import annotations

import argparse
import datetime

import config
from dataset import load_class_indices, make_tf_dataset, prepare_splits
from metrics import build_evaluation, save_metrics


def main():
    ap = argparse.ArgumentParser(description="Evaluasi model & tulis metrics.json")
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = ap.parse_args()

    import numpy as np
    import tensorflow as tf

    if not config.MODEL_PATH.exists() or not config.CLASS_INDICES_PATH.exists():
        raise SystemExit("Model/class_indices belum ada. Latih model dulu (train.py).")

    model = tf.keras.models.load_model(config.MODEL_PATH)
    class_indices = load_class_indices()
    classes = [l for l, _ in sorted(class_indices.items(), key=lambda kv: kv[1])]

    _, val_samples, test_samples, mode = prepare_splits()
    eval_samples = test_samples if test_samples else val_samples
    if not eval_samples:
        raise SystemExit("Tidak ada data evaluasi.")

    eval_ds = make_tf_dataset(eval_samples, class_indices, batch_size=args.batch_size)
    y_true = [class_indices[l] for _, l in eval_samples]
    probs = np.asarray(model.predict(eval_ds, verbose=0))
    y_pred = probs.argmax(axis=1).tolist()
    loss, _ = model.evaluate(eval_ds, verbose=0)

    metrics = build_evaluation(
        y_true,
        y_pred,
        classes,
        loss=loss,
        evaluated_at=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    save_metrics(metrics, config.METRICS_PATH)
    print(
        f"Akurasi: {metrics['accuracy']:.4f} | Macro-F1: {metrics['macro_f1']:.4f} "
        f"| sampel: {metrics['num_samples']} ({'Test' if test_samples else 'Val'})"
    )
    print(f"Metrics disimpan ke {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
