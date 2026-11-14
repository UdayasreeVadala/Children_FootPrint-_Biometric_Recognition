"""Training utilities for FootprintID.

Two training paths are supported:

1. **Dataset training** — trains MobileNetV2, EfficientNet-B0 and a custom CNN
   on ``Training_Data`` (or any dataset with the same layout).  Metrics, model
   files, confusion matrices and a registry are written under ``models/``.
2. **Enrollment retraining** — the original lightweight path used by the
   ``POST /api/retrain`` endpoint, which trains directly from the processed
   images in ``data/enrolled``.

Only the dataset path is imported by the command-line trainer.  The functions
that need TensorFlow import it lazily so the web application can still start
and serve static pages when the ML stack is unavailable.
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

try:
    import tensorflow as tf
except ModuleNotFoundError:  # Allows non-ML routes and tests before install.
    tf = None

from app.inference.architectures import (
    ARCHITECTURE_LABELS,
    build_classification_model,
    find_last_conv_layer,
    normalize_architecture_name,
    unfreeze_for_fine_tuning,
)
from app.inference.dataset import (
    PreprocessedSample,
    class_distribution,
    discover_dataset,
    grouped_split,
    load_manifest,
    preprocess_dataset,
    resolve_task,
    save_manifest,
    summarize_samples,
)

DEFAULT_ARCHITECTURES = ("mobilenetv2", "efficientnetb0", "custom_cnn")
DEFAULT_CACHE_DIRNAME = "dataset_224"
DEFAULT_MANIFEST_NAME = "dataset_manifest.json"


def _require_tensorflow() -> None:
    if tf is None:
        raise RuntimeError("TensorFlow is required for training. Run pip install -r requirements.txt.")


# ---------------------------------------------------------------------------
# Enrollment-based training (used by POST /api/retrain)
# ---------------------------------------------------------------------------

def load_enrollment_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Load processed enrollment PNGs, requiring two classes and two images each."""
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise ValueError("No enrolled children found. Enroll at least two children first.")

    with metadata_path.open("r", encoding="utf-8") as file:
        children = json.load(file).get("children", [])

    images, labels, usable_children = [], [], []
    for child in children:
        child_id = child.get("child_id")
        if not child_id:
            continue
        image_paths = sorted((data_dir / child_id).glob("*.png"))
        if len(image_paths) < 2:
            continue
        class_index = len(usable_children)
        usable_children.append(
            {"child_id": child_id, "child_name": child.get("child_name", child_id)}
        )
        for path in image_paths:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            images.append(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
            labels.append(class_index)

    if len(usable_children) < 2:
        raise ValueError("Retraining needs at least two children with two processed images each.")
    if len(images) < 4:
        raise ValueError("Retraining needs at least four usable processed images.")

    return np.asarray(images, dtype=np.float32), np.asarray(labels), usable_children


def build_training_model(num_classes: int, architecture: str = "custom_cnn"):
    """Build a classifier for enrollment retraining.

    The original function is kept for backward compatibility but now delegates
    to the shared architecture factory.
    """
    _require_tensorflow()
    architecture = normalize_architecture_name(architecture)
    weights = "imagenet" if architecture in {"mobilenetv2", "efficientnetb0"} else None
    try:
        return build_classification_model(architecture, num_classes, weights=weights, dropout=0.30)
    except Exception:
        # Offline environments may not have the ImageNet weights cached.
        return build_classification_model(architecture, num_classes, weights=None, dropout=0.30)


def _stratified_enrollment_split(
    labels: np.ndarray,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out a per-class validation slice so overfitting is measurable."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for class_index in np.unique(labels):
        indices = np.where(labels == class_index)[0]
        rng.shuffle(indices)
        if len(indices) >= 3:
            count = max(1, int(round(len(indices) * val_fraction)))
            count = min(count, len(indices) - 2)
        else:
            count = 0
        val_indices.extend(indices[:count].tolist())
        train_indices.extend(indices[count:].tolist())

    return (
        np.asarray(train_indices, dtype=np.int64),
        np.asarray(val_indices, dtype=np.int64),
    )


def _enrollment_augmentation_layers() -> "tf.keras.Sequential":
    """Geometric augmentation that keeps the background at exactly zero."""
    _require_tensorflow()
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.05, fill_mode="constant", fill_value=0.0),
            tf.keras.layers.RandomTranslation(0.10, 0.10, fill_mode="constant", fill_value=0.0),
            tf.keras.layers.RandomZoom(0.15, 0.15, fill_mode="constant", fill_value=0.0),
        ],
        name="enrollment_geometry_augmentation",
    )


def _random_erasing(images, probability: float = 0.4):
    """Blank a random rectangle to stop the head memorising fixed crease spots."""
    height = tf.shape(images)[0]
    width = tf.shape(images)[1]
    box_fraction = tf.random.uniform([], 0.08, 0.25)
    box_height = tf.cast(tf.cast(height, tf.float32) * box_fraction, tf.int32)
    box_width = tf.cast(tf.cast(width, tf.float32) * box_fraction, tf.int32)
    top = tf.random.uniform([], 0, tf.maximum(1, height - box_height), dtype=tf.int32)
    left = tf.random.uniform([], 0, tf.maximum(1, width - box_width), dtype=tf.int32)

    rows = tf.range(height)
    columns = tf.range(width)
    row_mask = (rows >= top) & (rows < top + box_height)
    column_mask = (columns >= left) & (columns < left + box_width)
    rectangle = tf.cast(row_mask[:, None] & column_mask[None, :], tf.float32)

    keep = tf.where(
        tf.random.uniform([]) < probability,
        1.0 - rectangle,
        tf.ones_like(rectangle),
    )
    return images * keep[..., None]


def _augment_enrollment_batch(images, labels, geometry):
    """Heavy augmentation for a single image (called from ``tf.data``).

    Gain, gamma and noise are all applied in a way that leaves the suppressed
    background at exactly ``0``, so the network still never sees background.
    """
    tensor = tf.cast(images, tf.float32)
    tensor = geometry(tensor, training=True)

    support = tf.cast(tensor > 1.0, tf.float32)

    # Multiplicative gain and gamma preserve a zero background.
    tensor = tensor * tf.random.uniform([], 0.85, 1.18)
    tensor = 255.0 * tf.pow(
        tf.clip_by_value(tensor / 255.0, 0.0, 1.0),
        tf.random.uniform([], 0.85, 1.15),
    )

    # Sensor-style noise, masked so the background cannot become grey.
    noise = tf.random.normal(tf.shape(tensor), mean=0.0, stddev=4.0)
    tensor = tensor + noise * support

    tensor = _random_erasing(tensor)
    tensor = tf.clip_by_value(tensor, 0.0, 255.0)
    return tensor, labels


def _enrollment_datasets(
    images: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    batch_size: int,
    seed: int,
):
    _require_tensorflow()
    train_images = images[train_indices]
    train_labels = labels[train_indices]
    if len(val_indices):
        val_images = images[val_indices]
        val_labels = labels[val_indices]
    else:
        # Tiny dataset: fall back to reporting the training slice as validation.
        val_images, val_labels = train_images, train_labels

    geometry = _enrollment_augmentation_layers()
    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        .shuffle(len(train_images), seed=seed, reshuffle_each_iteration=True)
        .map(
            lambda image, label: _augment_enrollment_batch(image, label, geometry),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    train_eval_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)
    return train_ds, val_ds, train_eval_ds, val_images, val_labels


def _enrollment_callbacks(models_dir: Path, model_path: Path, patience: int):
    """Checkpoint on validation loss and stop when it stops improving."""
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=max(2, patience // 3),
            min_lr=1e-6,
            verbose=0,
        ),
    ]


def retrain_from_enrollments(
    data_dir: Path,
    models_dir: Path,
    epochs: int = 40,
    architecture: str = "custom_cnn",
    *,
    fine_tune_epochs: int = 12,
    val_fraction: float = 0.2,
) -> dict:
    """Train a child classifier from the enrolled images.

    Overfitting is controlled with a per-class validation hold-out, heavy
    background-preserving augmentation, class weights, early stopping on
    validation loss, and an optional low-learning-rate fine-tuning phase.
    """
    _require_tensorflow()
    if not 1 <= epochs <= 200:
        raise ValueError("epochs must be between 1 and 200")
    if fine_tune_epochs < 0:
        raise ValueError("fine_tune_epochs cannot be negative")
    architecture = normalize_architecture_name(architecture)

    images, labels, children = load_enrollment_dataset(data_dir)
    class_count = len(children)
    tf.keras.utils.set_random_seed(42)

    train_indices, val_indices = _stratified_enrollment_split(
        labels, val_fraction=val_fraction
    )
    if len(train_indices) == 0:
        raise ValueError("The validation split left no training images.")

    batch_size = int(min(8, max(2, len(train_indices))))
    train_ds, val_ds, train_eval_ds, val_images, val_labels = _enrollment_datasets(
        images, labels, train_indices, val_indices, batch_size, seed=42
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "footprint_model.keras"
    classes_path = models_dir / "footprint_model_classes.json"
    meta_path = models_dir / "footprint_model_meta.json"

    model = build_training_model(class_count, architecture=architecture)
    base_trainable = False
    if architecture in {"mobilenetv2", "efficientnetb0"}:
        try:
            model.get_layer("backbone").trainable = False
            base_trainable = True
        except ValueError:
            base_trainable = False

    # Keras 2.15's SparseCategoricalCrossentropy has no label_smoothing
    # argument, and a custom smoothing loss would not deserialize in the
    # classifier's plain load_model() call. Overconfidence is instead handled by
    # the augmentation, validation-based early stopping and class weights.
    loss = "sparse_categorical_crossentropy"
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss,
        metrics=["accuracy"],
    )

    epochs_completed = 0
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=_enrollment_callbacks(models_dir, model_path, patience=10),
        class_weight=_class_weights(labels[train_indices], class_count),
        verbose=0,
    )
    epochs_completed += len(history.history.get("loss", []))

    fine_tune_completed = 0
    if base_trainable and fine_tune_epochs > 0:
        unfrozen = unfreeze_for_fine_tuning(model, number_of_layers=30)
        if unfrozen:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss=loss,
                metrics=["accuracy"],
            )
            fine_history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=fine_tune_epochs,
                callbacks=_enrollment_callbacks(models_dir, model_path, patience=6),
                class_weight=_class_weights(labels[train_indices], class_count),
                verbose=0,
            )
            fine_tune_completed = len(fine_history.history.get("loss", []))

    # Always persist the best checkpoint (lowest validation loss).
    if model_path.exists():
        model = tf.keras.models.load_model(model_path)

    train_loss, train_accuracy = model.evaluate(train_eval_ds, verbose=0)
    if len(val_indices):
        val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
    else:
        val_loss, val_accuracy = train_loss, train_accuracy

    model.save(model_path)
    with classes_path.open("w", encoding="utf-8") as file:
        json.dump(
            {str(index): child for index, child in enumerate(children)},
            file,
            indent=2,
        )
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "task": "identity",
                "architecture": architecture,
                "architecture_label": ARCHITECTURE_LABELS.get(architecture, architecture),
                "class_names": [child["child_id"] for child in children],
                "last_conv_layer": find_last_conv_layer(model),
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "source": "enrollments",
                "preprocessing": {
                    "pipeline": "ROI -> CLAHE -> Gabor -> background suppression",
                    "background_suppressed": True,
                },
                "augmentation": {
                    "rotation": 0.05,
                    "translation": 0.10,
                    "zoom": 0.15,
                    "gain": [0.85, 1.18],
                    "gamma": [0.85, 1.15],
                    "gaussian_noise": 4.0,
                    "random_erasing": 0.4,
                },
                "metrics": {
                    "train_accuracy": round(float(train_accuracy), 4),
                    "val_accuracy": round(float(val_accuracy), 4),
                    "overfit_gap": round(float(train_accuracy) - float(val_accuracy), 4),
                    "train_loss": round(float(train_loss), 4),
                    "val_loss": round(float(val_loss), 4),
                    "train_images": int(len(train_indices)),
                    "val_images": int(len(val_indices)),
                    "epochs_completed": int(epochs_completed),
                    "fine_tune_epochs_completed": int(fine_tune_completed),
                },
            },
            file,
            indent=2,
        )

    return {
        "model_path": str(model_path),
        "classes_path": str(classes_path),
        "architecture": architecture,
        "epochs_completed": int(epochs_completed),
        "fine_tune_epochs_completed": int(fine_tune_completed),
        "training_accuracy": round(float(train_accuracy) * 100, 2),
        "validation_accuracy": round(float(val_accuracy) * 100, 2),
        "overfit_gap": round(float(train_accuracy - val_accuracy) * 100, 2),
        "train_images": int(len(train_indices)),
        "val_images": int(len(val_indices)),
        "images_used": int(len(images)),
        "children_used": children,
    }


# ---------------------------------------------------------------------------
# Dataset training
# ---------------------------------------------------------------------------

def _load_cached_images(samples: list[PreprocessedSample]) -> tuple[np.ndarray, np.ndarray]:
    """Load cached grayscale PNGs into an RGB uint8 array."""
    images: list[np.ndarray] = []
    valid_samples: list[PreprocessedSample] = []
    for sample in samples:
        image = cv2.imread(sample.cache_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        if image.shape[:2] != (224, 224):
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        images.append(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
        valid_samples.append(sample)
    if not images:
        raise ValueError("None of the cached images could be loaded")
    return np.stack(images).astype(np.uint8), np.asarray(valid_samples)


def _labels_for(samples: list[PreprocessedSample], task: str, class_names: list[str]) -> np.ndarray:
    index = {name: position for position, name in enumerate(class_names)}
    return np.asarray([index[sample.label_for(task)] for sample in samples], dtype=np.int32)


def _make_augmentation() -> "tf.keras.Sequential":
    _require_tensorflow()
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.035, fill_mode="constant", fill_value=0.0),
            tf.keras.layers.RandomZoom(0.08, fill_mode="constant", fill_value=0.0),
            tf.keras.layers.RandomTranslation(0.04, 0.04, fill_mode="constant", fill_value=0.0),
        ],
        name="footprint_augmentation",
    )


def _augment_batch(images, labels, augmentation):
    images = tf.cast(images, tf.float32)
    images = augmentation(images, training=True)
    images = tf.image.random_brightness(images, max_delta=18.0)
    images = tf.image.random_contrast(images, lower=0.85, upper=1.15)
    images = tf.clip_by_value(images, 0.0, 255.0)
    return images, labels


def _make_datasets(
    train_samples: list[PreprocessedSample],
    validation_samples: list[PreprocessedSample],
    test_samples: list[PreprocessedSample],
    task: str,
    class_names: list[str],
    batch_size: int,
    seed: int,
):
    _require_tensorflow()
    train_images, train_valid = _load_cached_images(train_samples)
    val_images, val_valid = _load_cached_images(validation_samples)
    test_images, test_valid = _load_cached_images(test_samples)

    train_labels = _labels_for(train_valid, task, class_names)
    val_labels = _labels_for(val_valid, task, class_names)
    test_labels = _labels_for(test_valid, task, class_names)

    augmentation = _make_augmentation()
    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        .shuffle(buffer_size=min(len(train_images), 2048), seed=seed, reshuffle_each_iteration=True)
        .map(
            lambda images, labels: _augment_batch(images, labels, augmentation),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds, test_ds, test_images, test_labels


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> float:
    scores = []
    for index in range(class_count):
        true_positive = np.sum((y_true == index) & (y_pred == index))
        false_positive = np.sum((y_true != index) & (y_pred == index))
        false_negative = np.sum((y_true == index) & (y_pred != index))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)
    return float(np.mean(scores)) if scores else 0.0


def _binary_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    """Rank-based ROC-AUC for the positive class (no sklearn dependency)."""
    if len(np.unique(y_true)) != 2:
        return None
    scores = probabilities[:, 1].astype(np.float64)
    positive = y_true == 1
    negative = ~positive
    positive_count = int(positive.sum())
    negative_count = int(negative.sum())
    if positive_count == 0 or negative_count == 0:
        return None

    # Average ranks for ties.
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    index = 0
    while index < len(sorted_scores):
        end = index + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[index]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        ranks[order[index:end]] = average_rank
        index = end

    rank_sum = float(ranks[positive].sum())
    return float(
        (rank_sum - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def _save_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    destination: Path,
) -> None:
    """Render a readable confusion-matrix PNG without a plotting dependency."""
    size = max(360, 90 + 56 * len(class_names))
    cell_w = cell_h = 56
    image = np.full((size, size, 3), 18, dtype=np.uint8)
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            value = int(matrix[row, column])
            maximum = max(1, int(matrix.max()))
            intensity = int(30 + 180 * (value / maximum))
            color = (intensity, 60, 130 - intensity // 2)
            x0 = 120 + column * cell_w
            y0 = 90 + row * cell_h
            cv2.rectangle(image, (x0, y0), (x0 + cell_w, y0 + cell_h), color, -1)
            cv2.rectangle(image, (x0, y0), (x0 + cell_w, y0 + cell_h), (70, 70, 90), 1)
            cv2.putText(
                image,
                str(value),
                (x0 + 16, y0 + 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (240, 240, 250),
                2,
                cv2.LINE_AA,
            )
    for index, name in enumerate(class_names):
        cv2.putText(
            image,
            name[:14],
            (8, 90 + index * cell_h + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            name[:14],
            (120 + index * cell_w + 2, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (200, 200, 220),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(image, "Predicted", (120, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 212, 170), 2, cv2.LINE_AA)
    cv2.putText(image, "Actual", (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 212, 170), 2, cv2.LINE_AA)
    cv2.imwrite(str(destination), image)


def _save_history_plot(history: dict, destination: Path) -> None:
    """Draw training/validation accuracy and loss curves with OpenCV."""
    epochs = len(history.get("loss", []))
    if epochs == 0:
        return
    width, height = 720, 300
    image = np.full((height, width, 3), 18, dtype=np.uint8)
    left, right, top, bottom = 48, width - 20, 24, height - 36

    def to_points(values, color, maximum_value):
        points = []
        for index, value in enumerate(values):
            x = int(left + (right - left) * (index / max(1, epochs - 1)))
            y = int(bottom - (bottom - top) * (float(value) / max(maximum_value, 1e-6)))
            points.append((x, y))
        if len(points) > 1:
            cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(image, point, 2, color, -1)

    accuracy_values = list(history.get("accuracy", [])) + list(history.get("val_accuracy", []))
    loss_values = list(history.get("loss", [])) + list(history.get("val_loss", []))
    max_accuracy = max(accuracy_values + [0.01])
    max_loss = max(loss_values + [0.01])
    to_points(history.get("accuracy", []), (0, 212, 170), max_accuracy)
    to_points(history.get("val_accuracy", []), (123, 97, 255), max_accuracy)
    if max_loss > 0:
        # Draw loss on the lower half with its own scale.
        loss_scale_height = (bottom - top) * 0.42

        def to_loss_points(values, color):
            points = []
            for index, value in enumerate(values):
                x = int(left + (right - left) * (index / max(1, epochs - 1)))
                y = int(bottom - loss_scale_height * (float(value) / max_loss))
                points.append((x, y))
            if len(points) > 1:
                cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)

        to_loss_points(history.get("loss", []), (255, 71, 87))
        to_loss_points(history.get("val_loss", []), (255, 193, 7))

    cv2.rectangle(image, (left, top), (right, bottom), (60, 60, 80), 1)
    cv2.putText(image, "Accuracy (teal/purple) - Loss (red/amber)", (left, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 200), 1, cv2.LINE_AA)
    cv2.putText(image, f"Epochs: {epochs}", (right - 110, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 200), 1, cv2.LINE_AA)
    cv2.imwrite(str(destination), image)


def _evaluate(
    model,
    test_ds,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    class_names: list[str],
    destination_dir: Path,
    prefix: str = "model",
) -> dict:
    probabilities = model.predict(test_ds, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = float(np.mean(predictions == test_labels)) if len(test_labels) else 0.0
    top_three = float(
        np.mean(
            [
                int(true_label) in list(np.argsort(probabilities[index])[::-1][:3])
                for index, true_label in enumerate(test_labels)
            ]
        )
    ) if len(test_labels) else 0.0
    class_count = len(class_names)
    matrix = np.zeros((class_count, class_count), dtype=np.int32)
    for true_label, predicted_label in zip(test_labels, predictions):
        matrix[int(true_label), int(predicted_label)] += 1

    destination_dir.mkdir(parents=True, exist_ok=True)
    confusion_matrix_path = destination_dir / f"{prefix}_confusion_matrix.png"
    _save_confusion_matrix(matrix, class_names, confusion_matrix_path)

    per_class = []
    for index, name in enumerate(class_names):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "label": name,
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "support": int(matrix[index, :].sum()),
            }
        )

    return {
        "test_accuracy": round(accuracy, 4),
        "test_top3_accuracy": round(top_three, 4),
        "test_macro_f1": round(_macro_f1(test_labels, predictions, class_count), 4),
        "test_auc": _binary_auc(test_labels, probabilities),
        "test_images": int(len(test_labels)),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_path": str(confusion_matrix_path),
        "per_class": per_class,
        "class_distribution": {
            name: int(np.sum(test_labels == index))
            for index, name in enumerate(class_names)
        },
    }


def _class_weights(labels: np.ndarray, class_count: int) -> dict[int, float]:
    counts = np.bincount(labels, minlength=class_count).astype(np.float64)
    counts[counts == 0] = 1.0
    total = float(counts.sum())
    return {
        index: float(total / (class_count * count))
        for index, count in enumerate(counts)
    }


def _build_with_weights(architecture: str, class_count: int):
    """Build a model, falling back to random initialisation when offline."""
    if architecture in {"mobilenetv2", "efficientnetb0"}:
        try:
            return build_classification_model(
                architecture, class_count, weights="imagenet", dropout=0.30
            ), "imagenet"
        except Exception as error:  # pragma: no cover - depends on network/cache
            print(f"[warn] ImageNet weights unavailable for {architecture}: {error}")
    return build_classification_model(
        architecture, class_count, weights=None, dropout=0.30
    ), "scratch"


def _make_epoch_callback(
    progress: Callable[[dict], None] | None,
    architecture: str,
    phase: str,
    total_epochs: int,
):
    """Build a Keras callback that reports each finished epoch to the UI."""
    if progress is None:
        return None

    class EpochProgress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            metrics = {
                key: float(value)
                for key, value in (logs or {}).items()
                if isinstance(value, (int, float))
            }
            progress(
                {
                    "stage": "epoch",
                    "architecture": architecture,
                    "phase": phase,
                    "epoch": epoch + 1,
                    "total_epochs": total_epochs,
                    "logs": metrics,
                }
            )

    return EpochProgress()


def train_architecture(
    architecture: str,
    train_samples: list[PreprocessedSample],
    validation_samples: list[PreprocessedSample],
    test_samples: list[PreprocessedSample],
    task: str,
    class_names: list[str],
    models_dir: Path,
    *,
    epochs: int = 20,
    fine_tune_epochs: int = 8,
    batch_size: int = 16,
    seed: int = 42,
    verbose: int = 2,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Train one architecture and persist its model, metrics and plots."""
    _require_tensorflow()
    architecture = normalize_architecture_name(architecture)
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if fine_tune_epochs < 0:
        raise ValueError("fine_tune_epochs cannot be negative")

    models_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(seed)

    train_ds, val_ds, test_ds, test_images, test_labels = _make_datasets(
        train_samples,
        validation_samples,
        test_samples,
        task,
        class_names,
        batch_size,
        seed,
    )
    train_labels = _labels_for(train_samples, task, class_names)
    model, initialization = _build_with_weights(architecture, len(class_names))
    model_path = models_dir / f"{architecture}_{task}.keras"

    # Select checkpoints on validation loss rather than validation accuracy.
    # With a small, imbalanced dataset, accuracy can lock in a degenerate
    # majority-class model within the first epoch; loss keeps improving as the
    # classifier learns calibrated probabilities.
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=8,
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=0,
        ),
    ]

    started = time.time()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    head_callbacks = list(callbacks)
    epoch_callback = _make_epoch_callback(progress, architecture, "head", epochs)
    if epoch_callback is not None:
        head_callbacks.append(epoch_callback)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=head_callbacks,
        class_weight=_class_weights(train_labels, len(class_names)),
        verbose=verbose,
    )
    history_dict = {key: [float(value) for value in values] for key, values in history.history.items()}

    if architecture in {"mobilenetv2", "efficientnetb0"} and fine_tune_epochs > 0:
        unfrozen = unfreeze_for_fine_tuning(model, number_of_layers=30)
        if unfrozen:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            fine_callbacks = list(callbacks)
            fine_callback = _make_epoch_callback(
                progress, architecture, "fine_tune", fine_tune_epochs
            )
            if fine_callback is not None:
                fine_callbacks.append(fine_callback)
            fine_history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=fine_tune_epochs,
                callbacks=fine_callbacks,
                class_weight=_class_weights(train_labels, len(class_names)),
                verbose=verbose,
            )
            for key, values in fine_history.history.items():
                history_dict.setdefault(key, []).extend(float(value) for value in values)

    # Reload the best checkpoint before evaluating.
    if model_path.exists():
        model = tf.keras.models.load_model(model_path)

    metrics = _evaluate(
        model,
        test_ds,
        test_images,
        test_labels,
        class_names,
        models_dir,
        prefix=f"{architecture}_{task}",
    )
    metrics.update(
        {
            "architecture": architecture,
            "architecture_label": ARCHITECTURE_LABELS.get(architecture, architecture),
            "initialization": initialization,
            "epochs_requested": epochs,
            "fine_tune_epochs_requested": fine_tune_epochs,
            "epochs_completed": len(history_dict.get("loss", [])),
            "training_seconds": round(time.time() - started, 2),
            "model_path": str(model_path),
            "last_conv_layer": find_last_conv_layer(model),
            "parameters": int(model.count_params()),
        }
    )
    history_path = models_dir / f"{architecture}_{task}_history.json"
    training_plot_path = models_dir / f"{architecture}_{task}_training.png"
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history_dict, file, indent=2)
    _save_history_plot(history_dict, training_plot_path)
    metrics["history_path"] = str(history_path)
    metrics["training_plot_path"] = str(training_plot_path)
    with (models_dir / f"{architecture}_{task}_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return metrics


def _choose_best(metrics_by_architecture: dict[str, dict]) -> str:
    def score(item):
        metrics = item[1]
        return (
            float(metrics.get("test_accuracy") or 0.0),
            float(metrics.get("test_top3_accuracy") or 0.0),
            float(metrics.get("test_macro_f1") or 0.0),
        )

    return max(metrics_by_architecture.items(), key=score)[0]


def train_dataset_models(
    dataset_root: str | Path,
    models_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
    task: str = "auto",
    architectures: Iterable[str] = DEFAULT_ARCHITECTURES,
    epochs: int = 20,
    fine_tune_epochs: int = 8,
    batch_size: int = 16,
    seed: int = 42,
    force_preprocess: bool = False,
    max_images: int | None = None,
    include_sensors: Iterable[str] | None = ("Paper", "Scanner", "Camera"),
    verbose: int = 2,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Preprocess the dataset, train every requested architecture, and register it."""
    _require_tensorflow()
    dataset_root = Path(dataset_root)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir else dataset_root.parent / "data" / "cache" / DEFAULT_CACHE_DIRNAME
    manifest_path = cache_dir / DEFAULT_MANIFEST_NAME

    def report(event: dict) -> None:
        if progress:
            progress(event)

    architecture_list = [normalize_architecture_name(name) for name in architectures]
    samples = discover_dataset(dataset_root, include_sensors=include_sensors)
    if not samples:
        raise ValueError(f"No supported images found in {dataset_root}")
    resolved_task = resolve_task(samples, task)
    if resolved_task == "side":
        samples = [sample for sample in samples if sample.side != "Unknown"]
    if max_images:
        grouped: dict[str, list] = {}
        for sample in samples:
            grouped.setdefault(sample.label_for(resolved_task), []).append(sample)
        per_class = max(1, int(max_images) // max(1, len(grouped)))
        samples = [
            sample
            for label_samples in grouped.values()
            for sample in label_samples[:per_class]
        ]
    if not samples:
        raise ValueError("No samples remain after task-specific filtering")
    summary = summarize_samples(samples)

    report(
        {
            "stage": "discovered",
            "dataset": summary,
            "task": resolved_task,
            "cache_dir": str(cache_dir),
        }
    )

    if manifest_path.exists() and not force_preprocess:
        manifest = load_manifest(manifest_path)
        # The cache may contain all requested files already.
        if len(manifest) < len(samples):
            manifest = preprocess_dataset(
                samples,
                cache_dir,
                force=force_preprocess,
                progress=lambda index, total, detail: report(
                    {"stage": "preprocessing", "current": index, "total": total, "detail": str(detail)}
                ),
            )
            save_manifest(manifest, manifest_path)
    else:
        manifest = preprocess_dataset(
            samples,
            cache_dir,
            force=force_preprocess,
            progress=lambda index, total, detail: report(
                {"stage": "preprocessing", "current": index, "total": total, "detail": str(detail)}
            ),
        )
        save_manifest(manifest, manifest_path)

    if not manifest:
        raise ValueError("Preprocessing produced no usable images")

    class_names = sorted({sample.label_for(resolved_task) for sample in manifest})
    if len(class_names) < 2:
        raise ValueError(
            "Training needs at least two classes. The dataset contains only "
            f"one class for task '{resolved_task}'."
        )

    train_samples, validation_samples, test_samples = grouped_split(
        manifest, resolved_task, seed=seed
    )
    split_summary = {
        "train": class_distribution(train_samples, resolved_task),
        "validation": class_distribution(validation_samples, resolved_task),
        "test": class_distribution(test_samples, resolved_task),
    }
    report(
        {
            "stage": "split",
            "task": resolved_task,
            "class_names": class_names,
            "split": split_summary,
        }
    )

    metrics_by_architecture: dict[str, dict] = {}
    for index, architecture in enumerate(architecture_list):
        report(
            {
                "stage": "training",
                "architecture": architecture,
                "architecture_label": ARCHITECTURE_LABELS.get(architecture, architecture),
                "index": index,
                "total": len(architecture_list),
            }
        )
        def architecture_progress(event: dict, index: int = index) -> None:
            event = dict(event)
            event.setdefault("architecture", architecture)
            event["index"] = index
            event["architectures_total"] = len(architecture_list)
            event["architecture_label"] = ARCHITECTURE_LABELS.get(architecture, architecture)
            report(event)

        metrics = train_architecture(
            architecture,
            train_samples,
            validation_samples,
            test_samples,
            resolved_task,
            class_names,
            models_dir,
            epochs=epochs,
            fine_tune_epochs=fine_tune_epochs,
            batch_size=batch_size,
            seed=seed,
            verbose=verbose,
            progress=architecture_progress,
        )
        metrics_by_architecture[architecture] = metrics
        report(
            {
                "stage": "architecture_complete",
                "architecture": architecture,
                "architecture_label": ARCHITECTURE_LABELS.get(architecture, architecture),
                "index": index,
                "architectures_total": len(architecture_list),
                "metrics": metrics,
            }
        )

    best_architecture = _choose_best(metrics_by_architecture)
    best_metrics = metrics_by_architecture[best_architecture]
    best_model_path = Path(best_metrics["model_path"])

    # Compatibility aliases used by the existing application and tests.
    alias_model = models_dir / "footprint_model.keras"
    alias_classes = models_dir / "footprint_model_classes.json"
    alias_meta = models_dir / "footprint_model_meta.json"
    shutil.copy2(best_model_path, alias_model)
    with alias_classes.open("w", encoding="utf-8") as file:
        json.dump(
            {
                str(index): (
                    {"child_id": name, "child_name": name}
                    if resolved_task == "identity"
                    else name
                )
                for index, name in enumerate(class_names)
            },
            file,
            indent=2,
        )
    registry = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": resolved_task,
        "class_names": class_names,
        "dataset": summary,
        "split": split_summary,
        "architectures": metrics_by_architecture,
        "best_model": best_architecture,
        "active_model": best_architecture,
        "model_paths": {
            name: metrics["model_path"] for name, metrics in metrics_by_architecture.items()
        },
        "best_model_path": str(best_model_path),
        "calibration": {"center": 0.72, "scale": 0.08},
        "calibrations": {
            name: {"center": 0.72, "scale": 0.08} for name in metrics_by_architecture
        },
        "preprocessing": {
            "pipeline": "ROI (midfoot/arch 40-75%) -> CLAHE -> 6-orientation Gabor bank",
            "input_size": 224,
            "channels": "RGB (grayscale replicated)",
        },
    }
    registry_path = models_dir / "registry.json"
    with registry_path.open("w", encoding="utf-8") as file:
        json.dump(registry, file, indent=2)
    with alias_meta.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "task": resolved_task,
                "architecture": best_architecture,
                "architecture_label": ARCHITECTURE_LABELS.get(best_architecture, best_architecture),
                "class_names": class_names,
                "last_conv_layer": best_metrics.get("last_conv_layer", ""),
                "trained_at": registry["generated_at"],
                "source": "dataset",
                "registry": str(registry_path),
            },
            file,
            indent=2,
        )

    report(
        {
            "stage": "complete",
            "registry": str(registry_path),
            "best_model": best_architecture,
            "metrics": metrics_by_architecture,
            "class_names": class_names,
        }
    )
    return {
        "registry_path": str(registry_path),
        "best_model": best_architecture,
        "task": resolved_task,
        "class_names": class_names,
        "metrics": metrics_by_architecture,
        "dataset": summary,
        "split": split_summary,
    }


def load_registry(models_dir: str | Path) -> dict | None:
    path = Path(models_dir) / "registry.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def active_model_info(models_dir: str | Path) -> dict | None:
    """Return lightweight information about the active registered model."""
    registry = load_registry(models_dir)
    if not registry:
        return None
    active = registry.get("active_model") or registry.get("best_model")
    metrics = registry.get("architectures", {}).get(active, {})
    return {
        "architecture": active,
        "architecture_label": ARCHITECTURE_LABELS.get(active, active),
        "task": registry.get("task"),
        "class_names": registry.get("class_names", []),
        "best_model": registry.get("best_model"),
        "metrics": metrics,
        "model_path": metrics.get("model_path") or registry.get("best_model_path"),
    }
