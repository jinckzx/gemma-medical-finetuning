#!/usr/bin/env python3
"""
skin_training.py: DermNet skin disease classifier (save as skin_disease_image_model/training.py)

- Combines the dataset's train/test folders into one folder (no filename collisions)
- Builds tf.data pipelines with an 85/15 train/validation split (validation is NOT augmented)
- Two-phase training: (1) train the new head on a frozen base, (2) fine-tune the last N base layers
  at a low learning rate with BatchNorm layers kept frozen
- Saves weights + class_names.json + model_meta.json next to the weights file

Usage:
python training.py --source_dataset /content/data/dermnet --combined_dataset /content/combined_dataset \
    --epochs 30 --batch_size 16 --model efficientnet --unfreeze_layers 30 --weights /content/model.weights.h5
"""
import argparse
import json
import logging
import os
import shutil
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
AUTOTUNE = tf.data.AUTOTUNE


# ── Dataset preparation ────────────────────────────────────────────────────────
def count_images(dataset_folder: str) -> int:
    total = 0
    for _, _, files in os.walk(dataset_folder):
        total += len([f for f in files if f.lower().endswith(IMAGE_EXTS)])
    logger.info("Total images in %s: %d", dataset_folder, total)
    return total


def combine_dataset(source_dataset_path: str, combined_dataset_path: str, force: bool = False) -> None:
    """Merge <source>/train and <source>/test into one folder-per-class dataset."""
    if not force and os.path.isdir(combined_dataset_path) and count_images(combined_dataset_path) > 0:
        logger.info("Combined dataset already exists at %s, skipping (use --force_combine to rebuild).",
                    combined_dataset_path)
        return

    split_dirs = [(os.path.join(source_dataset_path, s), s + "_") for s in ("train", "test")]
    if not any(os.path.isdir(d) for d, _ in split_dirs):
        raise FileNotFoundError(f"No train/ or test/ folder found in {source_dataset_path}")

    os.makedirs(combined_dataset_path, exist_ok=True)
    for split_dir, prefix in split_dirs:
        if not os.path.isdir(split_dir):
            continue
        for class_name in sorted(os.listdir(split_dir)):
            src_class = os.path.join(split_dir, class_name)
            if not os.path.isdir(src_class):
                continue
            dst_class = os.path.join(combined_dataset_path, class_name)
            os.makedirs(dst_class, exist_ok=True)
            for filename in os.listdir(src_class):
                dst = os.path.join(dst_class, prefix + filename)  # prefix avoids train/test name clashes
                if not os.path.exists(dst):
                    shutil.copy(os.path.join(src_class, filename), dst)
    logger.info("Dataset combining complete.")


# ── Data pipeline ──────────────────────────────────────────────────────────────
def get_preprocess_fn(model_type: str):
    """EfficientNet has rescaling built in and expects raw 0-255 pixels."""
    if model_type == "efficientnet":
        return lambda x: x
    return tf.keras.applications.resnet50.preprocess_input


def make_datasets(dataset_dir: str, image_size: Tuple[int, int], batch_size: int,
                  val_split: float, seed: int, model_type: str):
    # shuffle=True with the SAME seed for both subsets keeps them disjoint
    common = dict(
        labels="inferred", label_mode="categorical", image_size=image_size,
        batch_size=batch_size, validation_split=val_split, seed=seed, shuffle=True,
    )
    train_raw = tf.keras.utils.image_dataset_from_directory(dataset_dir, subset="training", **common)
    val_raw = tf.keras.utils.image_dataset_from_directory(dataset_dir, subset="validation", **common)
    class_names: List[str] = list(train_raw.class_names)

    # class weights for imbalanced classes
    class_weights: Dict[int, float] = {}
    paths = getattr(train_raw, "file_paths", None)
    if paths:
        idx = {n: i for i, n in enumerate(class_names)}
        labels = np.array([idx[os.path.basename(os.path.dirname(p))] for p in paths])
        counts = np.bincount(labels, minlength=len(class_names)).astype(float)
        weights = len(labels) / (len(class_names) * np.maximum(counts, 1.0))
        class_weights = {i: float(w) for i, w in enumerate(weights)}
        logger.info("Class weights computed (min=%.3f, max=%.3f)", weights.min(), weights.max())

    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(30 / 360, fill_mode="nearest"),
        layers.RandomTranslation(0.2, 0.2, fill_mode="nearest"),
        layers.RandomZoom(0.3, fill_mode="nearest"),
        layers.RandomBrightness(0.2, value_range=(0, 255)),
    ])
    pre = get_preprocess_fn(model_type)

    def prep_train(x, y):
        return pre(augment(x, training=True)), y

    def prep_val(x, y):
        return pre(x), y

    skip_bad = tf.data.experimental.ignore_errors()  # skip unreadable images instead of crashing
    train_ds = train_raw.apply(skip_bad).map(prep_train, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_raw.apply(skip_bad).map(prep_val, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names, class_weights


# ── Model ──────────────────────────────────────────────────────────────────────
def build_model(num_classes: int, input_shape=(224, 224, 3), model_type: str = "resnet50"):
    """Returns (model, base_model). The base is fully frozen; use set_trainable() to unfreeze."""
    if model_type == "efficientnet":
        base_model = EfficientNetB3(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=base_model.input, outputs=outputs), base_model


def set_trainable(base_model, unfreeze_layers: int) -> None:
    """Unfreeze the last N base layers, but keep BatchNorm layers frozen (stable with small batches)."""
    for layer in base_model.layers:
        layer.trainable = False
    if unfreeze_layers <= 0:
        return
    unfrozen = 0
    for layer in base_model.layers[-unfreeze_layers:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
            unfrozen += 1
    logger.info("Unfroze %d layers (last %d of %d, BatchNorm kept frozen)",
                unfrozen, unfreeze_layers, len(base_model.layers))


def run_phase(model, train_ds, val_ds, epochs: int, lr: float, ckpt_path: str,
              patience: int, class_weights: Dict[int, float], name: str) -> float:
    logger.info("=== %s: %d epochs, lr=%g ===", name, epochs, lr)
    model.compile(optimizer=Adam(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(1, patience // 2),
                          min_lr=1e-7, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True,
                        save_weights_only=True, verbose=0),
    ]
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        class_weight=class_weights or None, callbacks=callbacks, verbose=2,
    )
    best = float(max(history.history["val_accuracy"]))
    if os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)  # make sure we continue from the best epoch
    logger.info("%s best val_accuracy: %.4f", name, best)
    return best


# ── Training ───────────────────────────────────────────────────────────────────
def train(dataset_dir: str, epochs: int, batch_size: int, output_weights: str,
          model_type: str, unfreeze_layers: int, patience: int,
          head_epochs: int, head_lr: float, finetune_lr: float,
          val_split: float, seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    logger.info("GPUs: %s", gpus if gpus else "NONE (training will be very slow)")

    image_size = (300, 300) if model_type == "efficientnet" else (224, 224)
    train_ds, val_ds, class_names, class_weights = make_datasets(
        dataset_dir, image_size, batch_size, val_split, seed, model_type)
    num_classes = len(class_names)
    logger.info("Num classes: %d", num_classes)

    model, base_model = build_model(num_classes, (image_size[0], image_size[1], 3), model_type)

    out_dir = os.path.dirname(os.path.abspath(output_weights))
    os.makedirs(out_dir, exist_ok=True)
    ckpt_head = os.path.join(out_dir, "ckpt_head.weights.h5")
    ckpt_ft = os.path.join(out_dir, "ckpt_finetune.weights.h5")

    results = []
    if unfreeze_layers > 0:
        head_epochs = max(1, min(head_epochs, epochs))
        ft_epochs = epochs - head_epochs
    else:
        head_epochs, ft_epochs = epochs, 0

    # Phase 1: train the new head on the frozen base
    best1 = run_phase(model, train_ds, val_ds, head_epochs, head_lr, ckpt_head,
                      min(patience, 3), class_weights, "Phase 1 (head only)")
    results.append((best1, ckpt_head))

    # Phase 2: fine-tune last N layers at a low learning rate
    if unfreeze_layers > 0 and ft_epochs > 0:
        set_trainable(base_model, unfreeze_layers)
        best2 = run_phase(model, train_ds, val_ds, ft_epochs, finetune_lr, ckpt_ft,
                          patience, class_weights, "Phase 2 (fine-tune)")
        results.append((best2, ckpt_ft))

    best_acc, best_path = max(results, key=lambda r: r[0])
    model.load_weights(best_path)
    loss, acc = model.evaluate(val_ds, verbose=0)
    logger.info("Final (best) model | val loss: %.4f | val accuracy: %.4f", loss, acc)

    model.save_weights(output_weights)
    logger.info("Weights saved to %s", output_weights)

    with open(os.path.join(out_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)
    meta = {
        "model": model_type,
        "image_size": list(image_size),
        "num_classes": num_classes,
        "head": "GlobalAveragePooling2D -> Dense(512, relu) -> Dropout(0.5) -> Dense(num_classes, softmax)",
        "preprocessing": ("raw RGB pixels in 0-255 float, no /255 (EfficientNet rescales internally)"
                          if model_type == "efficientnet"
                          else "tf.keras.applications.resnet50.preprocess_input"),
        "val_accuracy": float(acc),
    }
    with open(os.path.join(out_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved class_names.json and model_meta.json to %s", out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dataset", type=str, default=None,
                        help="Folder containing train/ and test/ (e.g. /content/data/dermnet)")
    parser.add_argument("--combined_dataset", type=str, default="./combined_dataset")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Folder to train on. Defaults to --combined_dataset")
    parser.add_argument("--force_combine", action="store_true")
    parser.add_argument("--epochs", type=int, default=30, help="Total epochs (head + fine-tune)")
    parser.add_argument("--head_epochs", type=int, default=5, help="Epochs for the head-only phase")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--weights", type=str, default="model.weights.h5")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "efficientnet"])
    parser.add_argument("--unfreeze_layers", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=1e-5)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.source_dataset:
        combine_dataset(args.source_dataset, args.combined_dataset, args.force_combine)

    # Train on the folder we just built (previously this fell back to ./combined_dataset by mistake)
    dataset_dir = args.dataset or args.combined_dataset
    if count_images(dataset_dir) == 0:
        raise FileNotFoundError(f"No images found in {dataset_dir}")

    train(
        dataset_dir=dataset_dir, epochs=args.epochs, batch_size=args.batch_size,
        output_weights=args.weights, model_type=args.model,
        unfreeze_layers=args.unfreeze_layers, patience=args.patience,
        head_epochs=args.head_epochs, head_lr=args.head_lr, finetune_lr=args.finetune_lr,
        val_split=args.val_split, seed=args.seed,
    )


if __name__ == "__main__":
    main()