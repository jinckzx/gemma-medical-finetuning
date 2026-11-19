
#!/usr/bin/env python3
"""
DermNet Training Pipeline (Production Ready)
--------------------------------------------
- Combines train/test folders into single dataset
- Counts dataset images
- Builds TF ImageDataGenerator with validation split
- Trains ResNet50 classifier (with EfficientNet option)
- Saves & loads weights

Usage:
python training.py     --source_dataset /data/dermnet     --combined_dataset ./combined_dataset     --epochs 30     --batch_size 16     --model efficientnet     --unfreeze_layers 30
"""
import os
import argparse
import logging
import shutil
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import ResNet50, EfficientNetB3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def combine_dataset(source_dataset_path: str, combined_dataset_path: str) -> None:
    """Combine train/test folders into a single dataset."""
    os.makedirs(combined_dataset_path, exist_ok=True)

    # Handle both "dermnet/train" and "dermnet" as source
    train_dir = os.path.join(source_dataset_path, "train")
    if not os.path.exists(train_dir):
        # Maybe the source is already the dermnet root with train/test
        train_dir = source_dataset_path

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found at {train_dir} or {source_dataset_path}/train")

    classes = os.listdir(train_dir)

    for class_name in classes:
        train_class_path = os.path.join(train_dir, class_name)
        test_class_path = os.path.join(source_dataset_path, "test", class_name)

        combined_class_path = os.path.join(combined_dataset_path, class_name)
        os.makedirs(combined_class_path, exist_ok=True)

        for folder_path in [train_class_path, test_class_path]:
            if not os.path.exists(folder_path):
                continue

            for filename in os.listdir(folder_path):
                src = os.path.join(folder_path, filename)
                dst = os.path.join(combined_class_path, filename)

                if not os.path.exists(dst):
                    shutil.copy(src, dst)

    logger.info("Dataset combining complete.")


def count_images(dataset_folder: str) -> int:
    total_images = 0

    for root, _, files in os.walk(dataset_folder):
        total_images += len([f for f in files if f.lower().endswith((".jpg", ".png", ".jpeg"))])

    logger.info("Total images: %d", total_images)
    return total_images


def build_generators(
    dataset_directory: str,
    image_size: Tuple[int, int],
    batch_size: int,
    validation_split: float = 0.15
):
    """Build improved data generators with enhanced augmentation."""
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.3,
        shear_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=validation_split,
    )

    train_gen = datagen.flow_from_directory(
        dataset_directory,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        subset="training",
    )

    val_gen = datagen.flow_from_directory(
        dataset_directory,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        subset="validation",
    )

    return train_gen, val_gen


def build_model(
    num_classes: int,
    input_shape=(224, 224, 3),
    model_type: str = "resnet50",
    unfreeze_layers: int = 30,
):
    """
    Build an improved classifier with optional base layer unfreezing.

    Args:
        num_classes: Number of output classes
        input_shape: Input image shape (H, W, C)
        model_type: "resnet50" or "efficientnet"
        unfreeze_layers: Number of layers to unfreeze from the end of the base model
    """
    if model_type == "efficientnet":
        base_model = EfficientNetB3(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
        )
    else:
        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
        )

    # Freeze base initially
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze last N layers for fine-tuning
    if unfreeze_layers > 0:
        total_layers = len(base_model.layers)
        start_unfreeze = max(0, total_layers - unfreeze_layers)
        for layer in base_model.layers[start_unfreeze:]:
            layer.trainable = True
        logger.info("Unfreezing %d layers (from layer %d to %d)",
                     unfreeze_layers, start_unfreeze, total_layers - 1)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train(
    dataset_dir: str,
    epochs: int,
    batch_size: int,
    output_weights: str,
    model_type: str = "resnet50",
    unfreeze_layers: int = 30,
    patience: int = 10,
):
    image_size = (224, 224)
    if model_type == "efficientnet":
        image_size = (300, 300)

    train_gen, val_gen = build_generators(
        dataset_dir,
        image_size=image_size,
        batch_size=batch_size
    )

    num_classes = train_gen.num_classes
    logger.info("Num classes: %d", num_classes)

    model = build_model(
        num_classes,
        input_shape=(image_size[0], image_size[1], 3),
        model_type=model_type,
        unfreeze_layers=unfreeze_layers,
    )

    # Compute class weights for imbalanced data
    import numpy as np
    from sklearn.utils import class_weight

    train_gen.reset()
    class_indices = train_gen.classes
    cw = class_weight.compute_class_weight(
        "balanced",
        classes=np.unique(class_indices),
        y=class_indices,
    )
    class_weights = dict(enumerate(cw))
    logger.info("Class weights computed (min=%.3f, max=%.3f)",
                 min(cw), max(cw))

    # Callbacks
    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            "best_model.weights.h5",
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // batch_size,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_gen.samples // batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # Save final weights
    model.save_weights(output_weights)
    logger.info("Weights saved to %s", output_weights)

    # Log final metrics
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    logger.info("Final train accuracy: %.4f | Val accuracy: %.4f",
                final_train_acc, final_val_acc)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_dataset", type=str, required=False)
    parser.add_argument("--combined_dataset", type=str, default="./combined_dataset")
    parser.add_argument("--dataset", type=str, default="./combined_dataset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--weights", type=str, default="model.weights.h5")
    parser.add_argument("--model", type=str, default="resnet50",
                        choices=["resnet50", "efficientnet"],
                        help="Base model architecture")
    parser.add_argument("--unfreeze_layers", type=int, default=30,
                        help="Number of base model layers to unfreeze")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")

    args = parser.parse_args()

    if args.source_dataset:
        combine_dataset(args.source_dataset, args.combined_dataset)

    count_images(args.dataset)

    train(
        dataset_dir=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_weights=args.weights,
        model_type=args.model,
        unfreeze_layers=args.unfreeze_layers,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
