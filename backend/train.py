from __future__ import annotations

import argparse
import datetime

import config
from dataset import (
    build_label_index,
    class_distribution,
    make_tf_dataset,
    prepare_splits,
    save_class_indices,
)
from metrics import build_evaluation, save_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Latih Car Brand Logo Classifier")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    p.add_argument("--min-images-per-class", type=int, default=config.MIN_IMAGES_PER_CLASS)
    p.add_argument("--fine-tune-at", type=int, default=config.FINE_TUNE_AT)
    p.add_argument("--fine-tune-epochs", type=int, default=config.FINE_TUNE_EPOCHS)
    p.add_argument("--val-split", type=float, default=config.VAL_SPLIT)
    return p.parse_args()


def build_model(num_classes, learning_rate, dropout=config.DROPOUT):
    import tensorflow as tf

    base = tf.keras.applications.MobileNetV2(
        input_shape=config.IMG_SIZE + (config.CHANNELS,),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=config.IMG_SIZE + (config.CHANNELS,))
    x = base(inputs, training=False)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=config.LABEL_SMOOTHING
        ),
        metrics=["accuracy"],
    )
    return model, base


def _callbacks():
    import tensorflow as tf

    return [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True, verbose=1
        ),
    ]


def main():
    args = parse_args()
    import numpy as np
    import tensorflow as tf

    tf.random.set_seed(config.SEED)

    print("[1/6] Menemukan & memuat dataset...")
    train_samples, val_samples, test_samples, mode = prepare_splits(
        config.IMAGES_DIR, args.min_images_per_class, args.val_split
    )
    if not train_samples:
        raise SystemExit("Tidak ada data latih. Periksa struktur folder dataset.")

    combined = train_samples + val_samples + test_samples
    class_indices = build_label_index(combined)
    classes = [l for l, _ in sorted(class_indices.items(), key=lambda kv: kv[1])]
    num_classes = len(classes)
    print(f"      Mode: {mode} | Kelas: {num_classes}")
    print(
        f"      Train={len(train_samples)} Val={len(val_samples)} "
        f"Test={len(test_samples)}"
    )
    print(f"      Distribusi train: {class_distribution(train_samples)}")

    print("[2/6] Menyimpan class_indices.json...")
    save_class_indices(class_indices)

    print("[3/6] Membangun tf.data pipeline...")
    train_ds = make_tf_dataset(
        train_samples, class_indices, batch_size=args.batch_size,
        shuffle=True, augment=True,
    )
    val_ds = (
        make_tf_dataset(val_samples, class_indices, batch_size=args.batch_size)
        if val_samples
        else None
    )
    test_ds = (
        make_tf_dataset(test_samples, class_indices, batch_size=args.batch_size)
        if test_samples
        else None
    )

    print("[4/6] Membangun model MobileNetV2...")
    model, base = build_model(num_classes, args.learning_rate)

    print("[5/6] Melatih classification head...")
    model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=_callbacks()
    )

    fine_tuned = bool(args.fine_tune_at and args.fine_tune_at > 0)
    if fine_tuned:
        print(f"      Fine-tuning {args.fine_tune_at} layer teratas...")
        base.trainable = True
        for layer in base.layers[: -args.fine_tune_at]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.learning_rate / 10),
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=config.LABEL_SMOOTHING
            ),
            metrics=["accuracy"],
        )
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=_callbacks(),
        )

    print("[6/6] Evaluasi, menulis metrics.json & menyimpan model...")
    eval_samples = test_samples if test_samples else val_samples
    eval_ds = test_ds if test_ds is not None else val_ds
    if eval_ds is not None and eval_samples:
        loss, acc = model.evaluate(eval_ds, verbose=0)
        tag = "Test" if test_samples else "Val"
        print(f"      {tag} accuracy: {acc:.4f} | {tag} loss: {loss:.4f}")

        probs = np.asarray(model.predict(eval_ds, verbose=0))
        y_pred = probs.argmax(axis=1).tolist()
        y_true = [class_indices[l] for _, l in eval_samples]
        metrics = build_evaluation(
            y_true,
            y_pred,
            classes,
            loss=loss,
            fine_tuned=fine_tuned,
            evaluated_at=datetime.datetime.now().isoformat(timespec="seconds"),
        )
        save_metrics(metrics, config.METRICS_PATH)
        print(f"      Metrics disimpan ke {config.METRICS_PATH}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(config.MODEL_PATH)
    print(f"      Model disimpan ke {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
