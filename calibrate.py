"""Calibrate embedding-similarity confidence for trained FootprintID models.

The bundled dataset contains a single child, so the models are trained on a
Left/Right foot-side task and the penultimate 256-D embedding is used for child
verification.  This script measures how similar embeddings are for:

* **genuine pairs** — same child, same foot side, different capture groups;
* **impostor pairs** — same child, opposite foot side.

With only one child available, opposite-side pairs are the closest available
proxy for impostor similarity.  The measured distributions set the logistic
calibration ``center`` and ``scale`` so the 70% rejection threshold lands
between the two distributions.

Run after training::

    python calibrate.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate embedding similarity confidence")
    parser.add_argument("--models-dir", default=str(BASE_DIR / "models"))
    parser.add_argument("--dataset", default=str(BASE_DIR / "Training_Data"))
    parser.add_argument("--manifest", default=str(BASE_DIR / "data" / "cache" / "dataset_224" / "dataset_manifest.json"))
    parser.add_argument("--architectures", default="", help="Comma-separated subset (default: all registered)")
    parser.add_argument("--include-phone", action="store_true")
    return parser.parse_args()


def load_embeddings(model_path: Path, samples: list) -> tuple[np.ndarray, list]:
    import tensorflow as tf

    from app.inference.architectures import build_embedding_model

    model = tf.keras.models.load_model(str(model_path))
    embedding_model = build_embedding_model(model)

    vectors = []
    valid_samples = []
    for sample in samples:
        image = cv2.imread(sample.cache_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        if image.shape[:2] != (224, 224):
            image = cv2.resize(image, (224, 224))
        vectors.append(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32))
        valid_samples.append(sample)
    if not vectors:
        raise ValueError("No cached images could be loaded")

    embeddings = embedding_model.predict(np.stack(vectors), batch_size=16, verbose=0)
    embeddings = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms, valid_samples


def split_groups(samples: list, embeddings: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return gallery/probe embeddings per side, split by capture group."""
    by_side_group: dict[str, dict[str, list[int]]] = {}
    for index, sample in enumerate(samples):
        by_side_group.setdefault(sample.side, {}).setdefault(sample.group, []).append(index)

    gallery: dict[str, np.ndarray] = {}
    probes: dict[str, np.ndarray] = {}
    for side, groups in by_side_group.items():
        ordered_groups = sorted(groups)
        if len(ordered_groups) < 2:
            indices = [index for group in ordered_groups for index in groups[group]]
            if len(indices) < 2:
                continue
            gallery[side] = embeddings[indices[::2]]
            probes[side] = embeddings[indices[1::2]]
            continue
        gallery_indices = [index for group in ordered_groups[::2] for index in groups[group]]
        probe_indices = [index for group in ordered_groups[1::2] for index in groups[group]]
        if gallery_indices and probe_indices:
            gallery[side] = embeddings[gallery_indices]
            probes[side] = embeddings[probe_indices]
    return gallery, probes


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.clip(left @ right.T, -1.0, 1.0)


def calibrate_samples(samples: list, embeddings: np.ndarray) -> dict:
    gallery, probes = split_groups(samples, embeddings)
    genuine: list[float] = []
    impostor: list[float] = []
    for side, probe_vectors in probes.items():
        if side not in gallery:
            continue
        genuine.extend(cosine_rows(probe_vectors, gallery[side]).ravel().tolist())
    side_names = list(gallery)
    for left_index, left_side in enumerate(side_names):
        for right_side in side_names[left_index + 1:]:
            if left_side in probes and right_side in gallery:
                impostor.extend(cosine_rows(probes[left_side], gallery[right_side]).ravel().tolist())
            if right_side in probes and left_side in gallery:
                impostor.extend(cosine_rows(probes[right_side], gallery[left_side]).ravel().tolist())

    if not genuine or not impostor:
        return {
            "center": 0.72,
            "scale": 0.08,
            "genuine_mean": None,
            "impostor_mean": None,
            "warning": "Not enough distinct capture groups to calibrate; using defaults.",
        }

    genuine_array = np.asarray(genuine)
    impostor_array = np.asarray(impostor)
    genuine_mean = float(genuine_array.mean())
    impostor_mean = float(impostor_array.mean())
    separation = genuine_mean - impostor_mean

    if separation <= 0.01:
        center, scale = 0.72, 0.08
        warning = "Genuine and impostor similarity overlap; using conservative defaults."
    else:
        center = (genuine_mean + impostor_mean) / 2.0
        # Place ~90% confidence at the genuine mean and ~10% at the impostor mean.
        scale = separation / (2.0 * math.log(9.0))
        scale = float(min(0.30, max(0.02, scale)))
        warning = None

    threshold_similarity = center + scale * math.log(0.70 / 0.30)
    far = float(np.mean(impostor_array >= threshold_similarity))
    frr = float(np.mean(genuine_array < threshold_similarity))

    result = {
        "center": round(float(center), 4),
        "scale": round(float(scale), 4),
        "genuine_mean": round(genuine_mean, 4),
        "impostor_mean": round(impostor_mean, 4),
        "separation": round(float(separation), 4),
        "threshold_similarity": round(float(threshold_similarity), 4),
        "false_accept_rate": round(far, 4),
        "false_reject_rate": round(frr, 4),
        "genuine_pairs": int(genuine_array.size),
        "impostor_pairs": int(impostor_array.size),
    }
    if warning:
        result["warning"] = warning
    return result


def main() -> int:
    args = parse_args()
    from app.inference.dataset import discover_dataset, load_manifest

    models_dir = Path(args.models_dir)
    registry_path = models_dir / "registry.json"
    if not registry_path.exists():
        raise SystemExit("No models/registry.json found. Run train.py first.")

    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    # Ensure every registered architecture has its own confusion-matrix image
    # and a recorded training-curve path.  The matrix itself is already stored
    # in the registry, so this does not require retraining.
    from app.inference.training import _save_confusion_matrix

    task_name = registry.get("task", "model")
    class_names = [str(name) for name in registry.get("class_names", [])]
    for architecture, metrics in registry.get("architectures", {}).items():
        matrix = metrics.get("confusion_matrix")
        if matrix and class_names and len(matrix) == len(class_names):
            matrix_path = models_dir / f"{architecture}_{task_name}_confusion_matrix.png"
            _save_confusion_matrix(np.asarray(matrix, dtype=np.int32), class_names, matrix_path)
            metrics["confusion_matrix_path"] = str(matrix_path)
        plot_path = models_dir / f"{architecture}_{task_name}_training.png"
        if plot_path.exists():
            metrics["training_plot_path"] = str(plot_path)

    architecture_paths = registry.get("model_paths", {})
    if not architecture_paths:
        architecture_paths = {
            key: value.get("model_path")
            for key, value in registry.get("architectures", {}).items()
        }
    selected = [name.strip() for name in args.architectures.split(",") if name.strip()]
    if selected:
        architecture_paths = {
            key: value for key, value in architecture_paths.items() if key in selected
        }

    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        samples = load_manifest(manifest_path)
        for sample in samples:
            sample.side = sample.side or "Unknown"
    else:
        sensors = ["Paper", "Scanner", "Camera"]
        if args.include_phone:
            sensors.append("Phone")
        discovered = discover_dataset(args.dataset, include_sensors=sensors)
        raise SystemExit(
            f"No preprocessing manifest at {manifest_path}. Run train.py first "
            f"({len(discovered)} source images were discovered)."
        )

    samples = [sample for sample in samples if sample.side in {"Left", "Right"}]
    if not samples:
        raise SystemExit("No Left/Right samples in the preprocessing manifest.")

    calibrations = {}
    report = {}
    for architecture, model_path in architecture_paths.items():
        if not model_path or not Path(model_path).exists():
            print(f"[skip] {architecture}: missing model file")
            continue
        print(f"[calibrate] {architecture} ...", flush=True)
        embeddings, valid_samples = load_embeddings(Path(model_path), samples)
        metrics = calibrate_samples(valid_samples, embeddings)
        calibrations[architecture] = {"center": metrics["center"], "scale": metrics["scale"]}
        report[architecture] = metrics
        print(
            f"  center={metrics['center']} scale={metrics['scale']} "
            f"genuine={metrics.get('genuine_mean')} impostor={metrics.get('impostor_mean')} "
            f"FAR={metrics.get('false_accept_rate')} FRR={metrics.get('false_reject_rate')}"
        )

    if not calibrations:
        raise SystemExit("No models could be calibrated.")

    registry["calibrations"] = calibrations
    active = registry.get("active_model") or registry.get("best_model")
    registry["calibration"] = calibrations.get(active, next(iter(calibrations.values())))
    registry["calibration_report"] = report
    registry["calibration_method"] = (
        "genuine = same side across capture groups; impostor = opposite side; "
        "logistic center/scale fitted between distributions"
    )
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    (models_dir / "calibration_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Updated {registry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
