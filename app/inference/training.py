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

import hashlib
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
    HEAD_LAYER_NAMES,
    build_classification_head_model,
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
    seen_hashes: set[str] = set()
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
            # Byte-identical repeats (re-enrolment of the same capture) only
            # double-weight a class and make the validation split misleading.
            digest = hashlib.sha1(image.tobytes()).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
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
    """Heavy geometric augmentation that keeps the background at exactly zero."""
    _require_tensorflow()
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.06, fill_mode="constant", fill_value=0.0),
            tf.keras.layers.RandomTranslation(0.10, 0.10, fill_mode="constant", fill_value=0.0),
            tf.keras.layers.RandomZoom(0.15, 0.15, fill_mode="constant", fill_value=0.0),
        ],
        name="enrollment_geometry_augmentation",
    )


def _single_random_erasing(images, probability: float):
    """Blank one random rectangle to stop the head memorising fixed crease spots."""
    height = tf.shape(images)[0]
    width = tf.shape(images)[1]
    box_fraction = tf.random.uniform([], 0.06, 0.22)
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


def _random_erasing(images, probability: float = 0.5, boxes: int = 2):
    """Apply several independent erasing boxes per sample."""
    for _ in range(max(1, boxes)):
        images = _single_random_erasing(images, probability)
    return images


def _augment_enrollment_batch(images, labels, geometry):
    """Heavy augmentation for a single image (called from ``tf.data``).

    Geometry, gain, gamma, noise, crease dropout and erasing are all applied in
    a way that leaves the suppressed background at exactly ``0``, so the
    network still never sees background.
    """
    tensor = tf.cast(images, tf.float32)
    tensor = geometry(tensor, training=True)

    support = tf.cast(tensor > 1.0, tf.float32)

    # Multiplicative gain and gamma preserve a zero background.
    tensor = tensor * tf.random.uniform([], 0.85, 1.20)
    tensor = 255.0 * tf.pow(
        tf.clip_by_value(tensor / 255.0, 0.0, 1.0),
        tf.random.uniform([], 0.85, 1.20),
    )

    # Sensor-style noise, masked so the background cannot become grey.
    noise = tf.random.normal(tf.shape(tensor), mean=0.0, stddev=5.0)
    tensor = tensor + noise * support

    # Random crease dropout: keep 90-100% of foreground pixels, never the
    # background (support == 0 forces the multiplier to 1 there).
    keep_probability = tf.random.uniform([], 0.90, 1.0)
    keep = tf.cast(tf.random.uniform(tf.shape(tensor)) < keep_probability, tf.float32)
    tensor = tensor * tf.maximum(keep, 1.0 - support)

    tensor = _random_erasing(tensor, probability=0.35, boxes=2)
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


def _ensure_native_keras_format() -> None:
    """Force native ``.keras`` saving for the current thread.

    Keras 2.15 picks the save format from a thread-local flag.  In a fresh
    worker thread the first save restores that flag to ``False``, so every
    later ``model.save("...keras")`` in the same thread silently falls back to
    legacy HDF5 (which also emits the "saving your model as an HDF5 file"
    warning).  ``/api/retrain`` runs in a FastAPI threadpool thread, so it hits
    exactly that case for every checkpoint after the first one.
    """
    try:
        from keras.src.saving import saving_lib
    except Exception:  # pragma: no cover - future Keras layouts
        return
    flag = getattr(saving_lib, "_SAVING_V3_ENABLED", None)
    if flag is not None:
        flag.value = True


def _enrollment_callbacks(
    models_dir: Path,
    model_path: Path,
    patience: int,
    monitor: str = "val_accuracy",
    mode: str = "max",
):
    """Checkpoint on validation accuracy and stop when it stops improving.

    Validation loss is a poor selection signal for this tiny enrolment set: it
    keeps falling while the model still confuses whole classes.  Selecting on
    accuracy checkpoints the model that actually recognises children.
    """
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=0.5,
            patience=max(2, patience // 3),
            min_lr=1e-6,
            verbose=0,
        ),
    ]


def retrain_from_enrollments(
    data_dir: Path,
    models_dir: Path,
    epochs: int = 150,
    architecture: str = "mobilenetv2",
    *,
    fine_tune_epochs: int = 12,
    fine_tune_learning_rate: float = 1e-5,
    val_fraction: float = 0.2,
) -> dict:
    """Train a child classifier from the enrolled images.

    Overfitting is controlled with a per-class validation hold-out, heavy
    background-preserving augmentation, class weights, early stopping on
    validation loss, and an optional low-learning-rate fine-tuning phase.
    """
    _require_tensorflow()
    _ensure_native_keras_format()
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
        callbacks=_enrollment_callbacks(models_dir, model_path, patience=20),
        class_weight=_class_weights(labels[train_indices], class_count),
        verbose=0,
    )
    epochs_completed += len(history.history.get("loss", []))

    fine_tune_completed = 0
    if base_trainable and fine_tune_epochs > 0:
        unfrozen = unfreeze_for_fine_tuning(model, number_of_layers=30)
        if unfrozen:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_learning_rate),
                loss=loss,
                metrics=["accuracy"],
            )
            fine_history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=fine_tune_epochs,
                callbacks=_enrollment_callbacks(models_dir, model_path, patience=12),
                class_weight=_class_weights(labels[train_indices], class_count),
                verbose=0,
            )
            fine_tune_completed = len(fine_history.history.get("loss", []))

    # Always persist the best checkpoint (highest validation accuracy).
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
                    "rotation": 0.06,
                    "translation": 0.10,
                    "zoom": 0.15,
                    "gain": [0.85, 1.20],
                    "gamma": [0.85, 1.20],
                    "gaussian_noise": 5.0,
                    "crease_dropout": [0.90, 1.0],
                    "random_erasing": {"probability": 0.35, "boxes": 2},
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
    # Crease locations carry the identity, so geometric augmentation stays
    # small: larger rotations/zooms move creases to where another subject's
    # creases are and turn a helpful regulariser into label noise.
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.02, fill_mode="constant", fill_value=0.0),
            tf.keras.layers.RandomZoom(0.06, fill_mode="constant", fill_value=0.0),
            tf.keras.layers.RandomTranslation(0.03, 0.03, fill_mode="constant", fill_value=0.0),
        ],
        name="footprint_augmentation",
    )


def _augment_batch(images, labels, augmentation):
    images = tf.cast(images, tf.float32)
    images = augmentation(images, training=True)
    images = tf.image.random_brightness(images, max_delta=12.0)
    images = tf.image.random_contrast(images, lower=0.90, upper=1.10)
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
    return train_ds, val_ds, test_ds, train_images, train_labels, val_images, val_labels, test_images, test_labels


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


def _draw_vertical_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw text rotated 90 degrees onto ``image`` at ``origin``."""
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    canvas = np.full((text_height + 10, text_width + 10, 3), 18, dtype=np.uint8)
    cv2.putText(canvas, text, (5, text_height + 4), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    rotated = np.ascontiguousarray(np.rot90(canvas))
    x, y = origin
    image[y:y + rotated.shape[0], x:x + rotated.shape[1]] = rotated


def _save_normalized_confusion_heatmap(
    matrix: np.ndarray,
    class_names: list[str],
    destination: Path,
) -> None:
    """Render a row-normalised confusion matrix as a colour heatmap PNG."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return
    class_count = matrix.shape[0]
    totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)
    labelled = class_count <= 12

    cell = 64 if labelled else max(3, 1100 // class_count)
    grid = cell * class_count
    left = 210 if labelled else 90
    top = 120
    bottom = 150 if labelled else 60
    width = left + grid + 200
    height = top + grid + bottom
    image = np.full((height, width, 3), 18, dtype=np.uint8)

    heat = cv2.applyColorMap(
        np.clip(normalised * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO
    )
    heat = cv2.resize(heat, (grid, grid), interpolation=cv2.INTER_NEAREST)
    image[top:top + grid, left:left + grid] = heat
    cv2.rectangle(image, (left, top), (left + grid, top + grid), (90, 90, 110), 2)

    if labelled:
        for row in range(class_count):
            for column in range(class_count):
                share = float(normalised[row, column])
                text = f"{share * 100:.0f}%"
                color = (25, 25, 32) if share > 0.55 else (245, 245, 250)
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.putText(
                    image,
                    text,
                    (
                        left + column * cell + (cell - text_width) // 2,
                        top + row * cell + (cell + text_height) // 2,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        for index, name in enumerate(class_names[:class_count]):
            label = str(name)[:14]
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(
                image,
                label,
                (left - 22 - text_width, top + index * cell + (cell + text_height) // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                label,
                (left + index * cell + (cell - text_width) // 2, top + grid + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 220),
                1,
                cv2.LINE_AA,
            )

    cv2.putText(image, "Normalized confusion matrix", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (215, 215, 235), 2, cv2.LINE_AA)
    cv2.putText(
        image,
        "Each row = actual class, cell = share of that class",
        (24, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (175, 175, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(image, "Predicted", (left, top - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 212, 170), 2, cv2.LINE_AA)
    actual_label = "Actual"
    (label_width, _), _ = cv2.getTextSize(actual_label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    _draw_vertical_text(
        image,
        actual_label,
        (18, top + grid // 2 - (label_width + 10) // 2),
        0.65,
        (0, 212, 170),
        2,
    )

    bar_x = left + grid + 46
    bar_width = 26
    gradient = np.linspace(1.0, 0.0, grid, dtype=np.float32) * 255.0
    bar = cv2.applyColorMap(
        np.repeat(gradient.astype(np.uint8).reshape(grid, 1), bar_width, axis=1), cv2.COLORMAP_INFERNO
    )
    image[top:top + grid, bar_x:bar_x + bar_width] = bar
    cv2.rectangle(image, (bar_x, top), (bar_x + bar_width, top + grid), (90, 90, 110), 2)
    for fraction, label in ((1.0, "100%"), (0.5, "50%"), (0.0, "0%")):
        y = int(round(top + grid * (1 - fraction)))
        cv2.line(image, (bar_x + bar_width, y), (bar_x + bar_width + 8, y), (90, 90, 110), 2)
        cv2.putText(
            image,
            label,
            (bar_x + bar_width + 14, y + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (175, 175, 200),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(image, "Row share", (bar_x - 10, top - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (175, 175, 200), 1, cv2.LINE_AA)

    cv2.imwrite(str(destination), image)


def _save_history_plot(history: dict, destination: Path) -> None:
    """Draw training/validation accuracy and loss curves with OpenCV."""
    epochs = len(history.get("loss", []))
    if epochs == 0:
        return
    width, height = 1600, 620
    image = np.full((height, width, 3), 18, dtype=np.uint8)
    top, bottom = 92, height - 96
    axis_color = (90, 90, 110)
    label_color = (175, 175, 200)
    grid_color = (40, 40, 54)

    accuracy_values = list(history.get("accuracy", [])) + list(history.get("val_accuracy", []))
    loss_values = list(history.get("loss", [])) + list(history.get("val_loss", []))
    max_accuracy = max(1.0, max(accuracy_values + [0.01]))
    max_loss = max(0.01, max(loss_values + [0.01]))

    if epochs <= 8:
        tick_indices = list(range(epochs))
    else:
        tick_indices = sorted({round(index * (epochs - 1) / 7) for index in range(8)})

    panels = (
        (
            "Accuracy",
            (120, 730),
            max_accuracy,
            True,
            (("accuracy", (0, 212, 170), "Train"), ("val_accuracy", (123, 97, 255), "Validation")),
        ),
        (
            "Loss",
            (880, 1560),
            max_loss,
            False,
            (("loss", (255, 71, 87), "Train"), ("val_loss", (255, 193, 7), "Validation")),
        ),
    )

    for title, (left, right), scale_max, is_percent, series in panels:
        def to_x(index: int) -> int:
            return int(round(left + (right - left) * (index / max(1, epochs - 1))))

        # Panel frame and title.
        cv2.rectangle(image, (left, top), (right, bottom), axis_color, 2)
        cv2.putText(image, title, (left, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (215, 215, 235), 2, cv2.LINE_AA)
        (title_width, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(
            image,
            f"{epochs} epochs",
            (left + title_width + 34, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            label_color,
            2,
            cv2.LINE_AA,
        )

        # Legend in the header line, right aligned with the panel.
        legend_widths = [cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] for _, _, name in series]
        legend_x = right - sum(width + 50 for width in legend_widths)
        for (_, color, name), text_width in zip(series, legend_widths):
            cv2.circle(image, (legend_x + 8, 42), 8, color, -1, cv2.LINE_AA)
            cv2.putText(
                image,
                name,
                (legend_x + 24, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                label_color,
                2,
                cv2.LINE_AA,
            )
            legend_x += text_width + 50

        # Y axis: ticks with value labels, plus light gridlines.
        for step in range(5):
            fraction = step / 4
            y = int(round(bottom - (bottom - top) * fraction))
            cv2.line(image, (left - 10, y), (left, y), axis_color, 2)
            cv2.line(image, (left, y), (right, y), grid_color, 1)
            value = scale_max * fraction
            text = f"{value * 100:.0f}%" if is_percent else f"{value:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(
                image,
                text,
                (left - 18 - text_width, y + text_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                label_color,
                2,
                cv2.LINE_AA,
            )

        # X axis: epoch ticks and axis title.
        for index in tick_indices:
            x = to_x(index)
            cv2.line(image, (x, bottom), (x, bottom + 10), axis_color, 2)
            text = str(index + 1)
            (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.putText(
                image,
                text,
                (x - text_width // 2, bottom + 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                label_color,
                2,
                cv2.LINE_AA,
            )
        (text_width, _), _ = cv2.getTextSize("Epoch", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(
            image,
            "Epoch",
            ((left + right) // 2 - text_width // 2, bottom + 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            label_color,
            2,
            cv2.LINE_AA,
        )

        # Curves for every series in this panel.
        for key, color, _ in series:
            points = [
                (
                    to_x(index),
                    int(round(bottom - (bottom - top) * (float(value) / max(scale_max, 1e-6)))),
                )
                for index, value in enumerate(history.get(key, []))
            ]
            if len(points) > 1:
                cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, color, 4, cv2.LINE_AA)
            for point in points:
                cv2.circle(image, point, 4, color, -1, cv2.LINE_AA)

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
    normalized_confusion_path = destination_dir / f"{prefix}_normalized_confusion.png"
    _save_normalized_confusion_heatmap(matrix, class_names, normalized_confusion_path)

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
        "normalized_confusion_path": str(normalized_confusion_path),
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


def export_augmented_dataset(
    data_dir: Path,
    output_dir: Path,
    copies_per_image: int = 10,
    *,
    seed: int = 42,
) -> dict:
    """Materialise augmented copies of the enrolled images on disk.

    Training itself uses on-the-fly augmentation (see ``_enrollment_datasets``),
    so this helper is for inspection, sharing or external tools.  Every copy is
    a fresh random variant produced by the exact same augmentation pipeline that
    training uses.
    """
    _require_tensorflow()
    if copies_per_image < 1:
        raise ValueError("copies_per_image must be at least 1")

    images, labels, children = load_enrollment_dataset(data_dir)
    tf.keras.utils.set_random_seed(seed)
    geometry = _enrollment_augmentation_layers()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    per_child: dict[str, int] = {}

    for class_index, child in enumerate(children):
        child_id = child["child_id"]
        child_dir = output_dir / child_id
        child_dir.mkdir(parents=True, exist_ok=True)
        for index in np.where(labels == class_index)[0]:
            source = tf.constant(images[index])
            for copy_index in range(copies_per_image):
                augmented, _ = _augment_enrollment_batch(
                    source, tf.constant(class_index, dtype=tf.int32), geometry
                )
                array = np.clip(augmented.numpy(), 0.0, 255.0).astype(np.uint8)
                path = child_dir / f"{child_id}_{int(index):03d}_aug{copy_index:02d}.png"
                if cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)):
                    written += 1
                    per_child[child_id] = per_child.get(child_id, 0) + 1

    return {
        "output_dir": str(output_dir),
        "children": len(children),
        "source_images": int(len(images)),
        "copies_per_image": int(copies_per_image),
        "images_written": written,
        "per_child": per_child,
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


def _footprint_backbone(model) -> "tf.keras.Model | None":
    """Return the pretrained backbone of a transfer model, if it has one."""
    base = getattr(model, "_footprint_base_model", None)
    if base is not None:
        return base
    try:
        return model.get_layer("backbone")
    except ValueError:
        return None


def _freeze_backbone(model) -> bool:
    """Freeze the pretrained backbone so the random head can learn safely."""
    base = _footprint_backbone(model)
    if base is None:
        return False
    base.trainable = False
    return True


def _backbone_feature_maps(model, images: np.ndarray, batch_size: int) -> np.ndarray:
    """Run the frozen backbone once over ``images`` and return its feature maps."""
    feature_map = getattr(model, "_footprint_feature_map", None)
    if feature_map is None:
        feature_map = _footprint_backbone(model)
    if feature_map is None:
        raise ValueError("This model has no pretrained backbone to cache.")
    extractor = tf.keras.Model(model.inputs, feature_map, name="backbone_features")
    return np.asarray(
        extractor.predict(np.asarray(images, dtype=np.float32), batch_size=batch_size, verbose=0)
    )


def _copy_head_weights(source: "tf.keras.Model", destination: "tf.keras.Model") -> int:
    """Copy the stage-1 head weights into the full model by layer name."""
    copied = 0
    for name in HEAD_LAYER_NAMES:
        try:
            source_layer = source.get_layer(name)
            destination_layer = destination.get_layer(name)
        except ValueError:
            continue
        weights = source_layer.get_weights()
        if not weights:
            continue  # Dropout layers carry no weights.
        if len(weights) != len(destination_layer.get_weights()):
            raise RuntimeError(
                f"Cannot copy '{name}': the head and the full model disagree on its parameters."
            )
        destination_layer.set_weights(weights)
        copied += 1
    if copied < 4:
        raise RuntimeError("Head pre-training did not produce the expected layers.")
    return copied


def _train_head_on_cached_features(
    model,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_images: np.ndarray,
    val_labels: np.ndarray,
    class_count: int,
    *,
    epochs: int,
    batch_size: int,
    verbose: int,
    callbacks: list,
):
    """Stage 1 of transfer training: fit the head on frozen backbone features.

    The backbone is run exactly once per split instead of once per step, which
    turns a 60 s/epoch job into a ~1 s/epoch job and makes a few dozen head
    epochs affordable on CPU.  The trained weights are copied back into the
    full model afterwards.
    """
    features_train = _backbone_feature_maps(model, train_images, batch_size)
    features_val = _backbone_feature_maps(model, val_images, batch_size)
    head = build_classification_head_model(
        features_train.shape[1:], class_count, dropout=0.30
    )
    head.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = head.fit(
        features_train,
        train_labels,
        validation_data=(features_val, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=_class_weights(train_labels, class_count),
        verbose=verbose,
    )
    return head, history


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
    head_epochs: int = 40,
    fine_tune_epochs: int = 8,
    batch_size: int = 16,
    seed: int = 42,
    verbose: int = 2,
    resume: bool = False,
    learning_rate: float | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Train one architecture and persist its model, metrics and plots.

    ``resume=True`` reloads the existing checkpoint for this architecture/task
    and continues training it.  Checkpoint loading is only supported for the
    custom CNN: the transfer models are trained through a frozen-backbone head
    stage and cannot be resumed halfway through it.
    """
    _require_tensorflow()
    _ensure_native_keras_format()
    architecture = normalize_architecture_name(architecture)
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if head_epochs < 1:
        raise ValueError("head_epochs must be at least 1")
    if fine_tune_epochs < 0:
        raise ValueError("fine_tune_epochs cannot be negative")

    if resume and architecture in {"mobilenetv2", "efficientnetb0"}:
        raise ValueError("Resuming is only supported for the custom CNN")

    models_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(seed)

    train_ds, val_ds, test_ds, train_images, train_labels, val_images, val_labels, test_images, test_labels = _make_datasets(
        train_samples,
        validation_samples,
        test_samples,
        task,
        class_names,
        batch_size,
        seed,
    )
    model_path = models_dir / f"{architecture}_{task}.keras"
    resumed = bool(resume and model_path.exists())
    if resumed:
        # Keep the pre-resume checkpoint: ModelCheckpoint starts a fresh
        # "best" comparison in this run, so the first epoch would overwrite it.
        model_path.with_name(model_path.name + ".prev").write_bytes(model_path.read_bytes())
        model = tf.keras.models.load_model(str(model_path))
        initialization = (
            "scratch-resumed" if architecture == "custom_cnn" else "imagenet-resumed"
        )
    else:
        model, initialization = _build_with_weights(architecture, len(class_names))

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

    history_path = models_dir / f"{architecture}_{task}_history.json"
    previous_history: dict[str, list[float]] = {}
    if resumed and history_path.exists():
        try:
            loaded_history = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(loaded_history, dict):
                previous_history = {
                    str(key): [float(value) for value in values]
                    for key, values in loaded_history.items()
                    if isinstance(values, list)
                }
        except (OSError, json.JSONDecodeError, ValueError):
            previous_history = {}

    started = time.time()
    history_dict: dict[str, list[float]] = {}
    head_epochs_completed = 0
    transfer = architecture in {"mobilenetv2", "efficientnetb0"}
    frozen = transfer and _freeze_backbone(model)

    if frozen:
        # Stage 1: feature extraction.  The ImageNet backbone stays frozen and
        # its feature maps are cached, so the head gets dozens of epochs for the
        # price of a couple of end-to-end ones.  Training the whole backbone at
        # the head learning rate (1e-3) destroys the pretrained filters and
        # leaves the model at chance on 220 identities.
        cache_callbacks = [
            callback for callback in callbacks
            if not isinstance(callback, tf.keras.callbacks.ModelCheckpoint)
        ]
        epoch_callback = _make_epoch_callback(
            progress, architecture, "head", head_epochs
        )
        if epoch_callback is not None:
            cache_callbacks.append(epoch_callback)
        head, history = _train_head_on_cached_features(
            model,
            train_images,
            train_labels,
            val_images,
            val_labels,
            len(class_names),
            epochs=head_epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=cache_callbacks,
        )
        _copy_head_weights(head, model)
        history_dict = {
            key: [float(value) for value in values] for key, values in history.history.items()
        }
        head_epochs_completed = len(history_dict.get("loss", []))
        model.save(model_path)
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate if learning_rate is not None else 1e-3
            ),
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
        history_dict = {
            key: [float(value) for value in values] for key, values in history.history.items()
        }

    if frozen and fine_tune_epochs > 0:
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

    if previous_history:
        merged_history = {key: list(values) for key, values in previous_history.items()}
        for key, values in history_dict.items():
            merged_history.setdefault(key, []).extend(values)
        history_dict = merged_history

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
            "head_epochs_requested": head_epochs,
            "head_epochs_completed": head_epochs_completed,
            "frozen_backbone": bool(frozen),
            "fine_tune_epochs_requested": fine_tune_epochs,
            "epochs_completed": len(history_dict.get("loss", [])),
            "resumed": resumed,
            "training_seconds": round(time.time() - started, 2),
            "model_path": str(model_path),
            "last_conv_layer": find_last_conv_layer(model),
            "parameters": int(model.count_params()),
        }
    )
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
    head_epochs: int = 40,
    fine_tune_epochs: int = 8,
    batch_size: int = 16,
    seed: int = 42,
    force_preprocess: bool = False,
    max_images: int | None = None,
    include_sensors: Iterable[str] | None = ("Paper", "Scanner", "Camera"),
    verbose: int = 2,
    resume: bool = False,
    learning_rate: float | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Preprocess the dataset, train every requested architecture, and register it.

    ``resume=True`` continues every requested architecture from its saved
    checkpoint (custom CNN only) and keeps the other registered architectures
    in the comparison instead of replacing the registry with this run alone.
    """
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

    previous_architectures: dict[str, dict] = {}
    if resume:
        previous_registry_path = models_dir / "registry.json"
        if previous_registry_path.exists():
            try:
                with previous_registry_path.open("r", encoding="utf-8") as file:
                    previous_registry = json.load(file)
            except (OSError, json.JSONDecodeError):
                previous_registry = {}
            if (
                isinstance(previous_registry, dict)
                and previous_registry.get("task") == resolved_task
            ):
                previous_architectures = dict(
                    previous_registry.get("architectures") or {}
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
            head_epochs=head_epochs,
            fine_tune_epochs=fine_tune_epochs,
            batch_size=batch_size,
            seed=seed,
            verbose=verbose,
            resume=resume,
            learning_rate=learning_rate,
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

    if previous_architectures:
        # A resumed custom-CNN run must not delete the transfer models that are
        # already registered for this task.
        merged_architectures = dict(previous_architectures)
        merged_architectures.update(metrics_by_architecture)
        metrics_by_architecture = merged_architectures

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
