"""Classification, confidence scoring, and unknown-child rejection.

Two inference modes are supported:

* **Softmax mode** — a model trained on two or more child identities returns a
  probability for every enrolled child.  The highest probability is the
  confidence score and scores below the configured threshold are rejected.
* **Embedding verification mode** — when the available dataset contains only
  one child, a child classifier cannot be trained.  The model's 256-D
  embedding is compared with the mean embedding of every enrolled child using
  cosine similarity.  Similarities are calibrated to a 0–100% confidence and
  the same rejection threshold is applied.

The web application always presents the mode and the raw similarity so a
similarity-based result is never mistaken for a softmax probability.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    import tensorflow as tf
except ModuleNotFoundError:  # Allows non-ML routes and demo tests before install.
    tf = None

from app.inference.architectures import (
    ARCHITECTURE_LABELS,
    build_classification_model,
    build_embedding_model,
    find_last_conv_layer,
)


class FootprintClassifier:
    """Identify an enrolled child from a preprocessed footprint."""

    def __init__(
        self,
        model_path: str | None = None,
        confidence_threshold: float = 0.70,
        models_dir: str | Path | None = None,
        enrollment_dir: str | Path | None = None,
    ):
        if not 0.0 < confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be greater than 0 and at most 1")

        self.confidence_threshold = confidence_threshold
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).resolve().parents[2] / "models"
        self.enrollment_dir = (
            Path(enrollment_dir)
            if enrollment_dir
            else Path(__file__).resolve().parents[2] / "data" / "enrolled"
        )

        self.model: tf.keras.Model | None = None
        self.embedding_model: tf.keras.Model | None = None
        self.class_records: dict[int, dict[str, str]] = {}
        self.is_demo = True
        self.last_conv_layer_name = ""
        self.architecture = "demo"
        self.architecture_label = "Demo MobileNetV2"
        self.task = "demo"
        self.match_mode = "demo"
        self.model_path: str | None = None
        self.registry: dict | None = None
        self.model_metrics: dict = {}
        self.calibration = {"center": 0.72, "scale": 0.08}
        self.class_names: list[str] = []

        self._enrollment_cache_key: tuple | None = None
        self._child_embeddings: dict[str, np.ndarray] = {}
        self._child_records: dict[str, dict[str, str]] = {}

        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        elif not self._load_registered_model():
            self._load_legacy_model()
        if self.model is None:
            self._create_demo_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _read_json(self, path: Path) -> dict | list | None:
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except (OSError, json.JSONDecodeError):
            return None

    def _load_registered_model(self) -> bool:
        registry = self._read_json(self.models_dir / "registry.json")
        if not isinstance(registry, dict):
            return False
        active = registry.get("active_model") or registry.get("best_model")
        model_paths = registry.get("model_paths", {})
        candidate = model_paths.get(active)
        if not candidate:
            candidate = registry.get("best_model_path")
        if not candidate or not Path(candidate).exists():
            return False

        self.registry = registry
        self.task = str(registry.get("task", "identity"))
        self.class_names = [str(name) for name in registry.get("class_names", [])]
        self.architecture = str(active or registry.get("best_model") or "unknown")
        calibration = registry.get("calibration", {}) or {}
        per_architecture = (registry.get("calibrations", {}) or {}).get(self.architecture)
        if isinstance(per_architecture, dict):
            calibration = per_architecture
        self.calibration = {
            "center": float(calibration.get("center", 0.72)),
            "scale": float(calibration.get("scale", 0.08)),
        }
        self.architecture_label = ARCHITECTURE_LABELS.get(self.architecture, self.architecture)
        self.model_metrics = registry.get("architectures", {}).get(self.architecture, {})
        self.load_model(candidate, task=self.task, class_names=self.class_names)
        return True

    def _load_legacy_model(self) -> bool:
        for extension in (".keras", ".h5"):
            candidate = self.models_dir / f"footprint_model{extension}"
            if candidate.exists():
                self.load_model(str(candidate))
                return True
        return False

    def _create_demo_model(self) -> None:
        """Create an untrained MobileNetV2 architecture for demo/Grad-CAM use."""
        if tf is None:
            self.model = None
            return
        tf.keras.utils.set_random_seed(42)
        self.model = build_classification_model("mobilenetv2", 5, weights=None, dropout=0.2)
        self.last_conv_layer_name = find_last_conv_layer(self.model)
        self.is_demo = True
        self.task = "demo"
        self.match_mode = "demo"
        self.architecture = "demo"
        self.architecture_label = "Demo MobileNetV2"
        self.embedding_model = build_embedding_model(self.model)

    def load_model(
        self,
        model_path: str,
        *,
        task: str | None = None,
        class_names: list[str] | None = None,
        meta_path: str | None = None,
        classes_path: str | None = None,
    ) -> None:
        """Load a trained Keras model and its optional JSON sidecars."""
        if tf is None:
            raise RuntimeError(
                "TensorFlow is required to load a trained model. "
                "Run pip install -r requirements.txt."
            )
        path = Path(model_path)
        self.model = tf.keras.models.load_model(str(path))
        self.model_path = str(path)
        self.is_demo = False
        self.last_conv_layer_name = find_last_conv_layer(self.model)
        try:
            self.embedding_model = build_embedding_model(self.model)
        except ValueError:
            self.embedding_model = None

        sidecar_meta = self._read_json(path.with_name(f"{path.stem}_meta.json"))
        if meta_path:
            sidecar_meta = self._read_json(Path(meta_path)) or sidecar_meta
        sidecar_classes = self._read_json(path.with_name(f"{path.stem}_classes.json"))
        if classes_path:
            sidecar_classes = self._read_json(Path(classes_path)) or sidecar_classes

        resolved_task = task
        resolved_names = class_names
        if isinstance(sidecar_meta, dict):
            resolved_task = resolved_task or sidecar_meta.get("task")
            resolved_names = resolved_names or sidecar_meta.get("class_names")
            if not self.last_conv_layer_name:
                self.last_conv_layer_name = str(sidecar_meta.get("last_conv_layer", ""))
            self.architecture = str(sidecar_meta.get("architecture", self.architecture))
            self.architecture_label = str(
                sidecar_meta.get("architecture_label", ARCHITECTURE_LABELS.get(self.architecture, self.architecture))
            )

        if sidecar_classes is not None:
            self.update_class_records(sidecar_classes)
        if resolved_names:
            self.class_names = [str(name) for name in resolved_names]
            if not self.class_records:
                self.update_class_records(
                    {str(index): name for index, name in enumerate(self.class_names)}
                )

        self.task = str(resolved_task or ("identity" if self.class_records else "embedding"))
        if self.task in {"identity", "visit"} and self.class_records:
            self.match_mode = "softmax"
        else:
            self.match_mode = "embedding"
        if not self.architecture_label:
            self.architecture_label = ARCHITECTURE_LABELS.get(self.architecture, self.architecture)

    @staticmethod
    def _find_last_conv_layer(model: tf.keras.Model) -> str:
        return find_last_conv_layer(model)

    # ------------------------------------------------------------------
    # Class labels
    # ------------------------------------------------------------------
    def update_class_records(self, records: dict | list) -> None:
        """Set model-index labels from metadata or a saved class JSON file."""
        normalized: dict[int, dict[str, str]] = {}
        iterable = records.items() if isinstance(records, dict) else enumerate(records)
        for index, value in iterable:
            if isinstance(value, dict):
                child_id = str(value.get("child_id", value.get("id", index)))
                child_name = str(value.get("child_name", value.get("name", child_id)))
            else:
                child_id = str(value)
                child_name = str(value)
            normalized[int(index)] = {"child_id": child_id, "child_name": child_name}
        self.class_records = normalized

    def update_class_names(self, class_names: dict) -> None:
        """Backward-compatible shorthand for callers that pass names only."""
        self.update_class_records(class_names)

    # ------------------------------------------------------------------
    # Image preparation
    # ------------------------------------------------------------------
    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum()

    @staticmethod
    def _as_rgb_224(image: np.ndarray) -> np.ndarray:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("A non-empty image is required for classification")
        if image.shape[:2] != (224, 224):
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        else:
            image = image.copy()
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.ndim == 3 and image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raise ValueError("Image must be grayscale or have three channels")

    def preprocess_for_model(self, image: np.ndarray) -> np.ndarray:
        """Convert grayscale/BGR input to a raw 0–255 RGB batch of 224×224."""
        rgb = self._as_rgb_224(image)
        return np.expand_dims(rgb.astype(np.float32), axis=0)

    # ------------------------------------------------------------------
    # Demo probabilities
    # ------------------------------------------------------------------
    def _demo_probabilities(self, image: np.ndarray) -> np.ndarray:
        """Make repeatable simulated probabilities for the enrolled class count."""
        class_count = max(len(self.class_records), 1)
        digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        rng = np.random.default_rng(seed)
        logits = rng.normal(size=class_count)
        logits[int(seed % class_count)] += 2.0
        return self._softmax(logits)

    def _model_probabilities(self, image: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("No model is loaded")
        raw = np.asarray(
            self.model.predict(self.preprocess_for_model(image), verbose=0)[0],
            dtype=np.float64,
        )
        if raw.ndim != 1 or raw.size == 0:
            raise ValueError("The model returned an invalid prediction shape")
        if np.any(raw < 0) or not np.isclose(raw.sum(), 1.0, atol=1e-4):
            return self._softmax(raw)
        return raw / raw.sum()

    # ------------------------------------------------------------------
    # Enrollment embedding index
    # ------------------------------------------------------------------
    def _enrollment_images(self) -> dict[str, list[Path]]:
        metadata = self._read_json(self.enrollment_dir / "metadata.json")
        children: dict[str, list[Path]] = {}
        if isinstance(metadata, dict):
            for child in metadata.get("children", []):
                child_id = str(child.get("child_id", "")).strip()
                if not child_id:
                    continue
                directory = self.enrollment_dir / child_id
                if directory.is_dir():
                    images = sorted(directory.glob("*.png"))
                    if images:
                        children[child_id] = images
        if not children:
            # Metadata might be missing while folders still exist.
            for directory in sorted(self.enrollment_dir.glob("*")):
                if directory.is_dir():
                    images = sorted(directory.glob("*.png"))
                    if images:
                        children[directory.name] = images
        return children

    def _enrollment_metadata(self) -> dict[str, str]:
        metadata = self._read_json(self.enrollment_dir / "metadata.json")
        names: dict[str, str] = {}
        if isinstance(metadata, dict):
            for child in metadata.get("children", []):
                child_id = str(child.get("child_id", "")).strip()
                if child_id:
                    names[child_id] = str(child.get("child_name", child_id))
        return names

    def _current_enrollment_cache_key(self, images: dict[str, list[Path]]) -> tuple:
        metadata_path = self.enrollment_dir / "metadata.json"
        metadata_stamp = metadata_path.stat().st_mtime_ns if metadata_path.exists() else 0
        image_stamp = tuple(
            (child_id, str(path), path.stat().st_mtime_ns)
            for child_id, paths in sorted(images.items())
            for path in paths
        )
        return (self.model_path, metadata_stamp, image_stamp)

    def refresh_enrollment_index(self, force: bool = False) -> None:
        """Compute mean embeddings for every enrolled child (cached by mtime)."""
        if self.embedding_model is None:
            self._child_embeddings = {}
            self._child_records = {}
            return
        images = self._enrollment_images()
        cache_key = self._current_enrollment_cache_key(images)
        if not force and cache_key == self._enrollment_cache_key:
            return

        names = self._enrollment_metadata()
        child_embeddings: dict[str, np.ndarray] = {}
        for child_id, paths in images.items():
            vectors: list[np.ndarray] = []
            batch: list[np.ndarray] = []
            for path in paths:
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                batch.append(image)
                if len(batch) >= 8:
                    vectors.extend(self._embed_images(batch))
                    batch = []
            if batch:
                vectors.extend(self._embed_images(batch))
            if vectors:
                mean = np.mean(np.stack(vectors), axis=0)
                norm = np.linalg.norm(mean)
                if norm > 0:
                    child_embeddings[child_id] = mean / norm

        self._child_embeddings = child_embeddings
        self._child_records = {
            child_id: {"child_id": child_id, "child_name": names.get(child_id, child_id)}
            for child_id in child_embeddings
        }
        self._enrollment_cache_key = cache_key

    def _embed_images(self, images: Iterable[np.ndarray]) -> list[np.ndarray]:
        if self.embedding_model is None:
            return []
        batch = np.stack(
            [self._as_rgb_224(image).astype(np.float32) for image in images],
            axis=0,
        )
        vectors = np.asarray(self.embedding_model.predict(batch, verbose=0), dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return list(vectors / norms)

    def _query_embedding(self, image: np.ndarray) -> np.ndarray:
        vectors = self._embed_images([image])
        if not vectors:
            raise RuntimeError("The loaded model does not expose an embedding layer")
        return vectors[0]

    @staticmethod
    def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator == 0:
            return 0.0
        return float(np.dot(left, right) / denominator)

    def _similarity_confidence(self, similarity: float) -> float:
        center = float(self.calibration.get("center", 0.72))
        scale = max(1e-4, float(self.calibration.get("scale", 0.08)))
        value = (similarity - center) / scale
        # Numerically stable logistic function.
        if value >= 0:
            return float(1.0 / (1.0 + math.exp(-value)))
        exponential = math.exp(value)
        return float(exponential / (1.0 + exponential))

    def get_child_embedding(self, child_id: str | None) -> np.ndarray | None:
        if not child_id:
            return None
        self.refresh_enrollment_index()
        return self._child_embeddings.get(child_id)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, image: np.ndarray) -> dict:
        """Return top-three scores and reject predictions below the threshold."""
        if self.is_demo:
            return self._predict_demo(image)
        if self.match_mode == "embedding":
            return self._predict_embedding(image)
        return self._predict_softmax(image)

    def _label_for(self, index: int) -> dict[str, str]:
        return self.class_records.get(
            index, {"child_id": f"Class_{index}", "child_name": f"Class {index}"}
        )

    def _base_result(self) -> dict:
        # ``getattr`` keeps the small unit-test fixtures that construct the
        # classifier with ``__new__`` backwards compatible.
        architecture = getattr(self, "architecture", "demo")
        return {
            "threshold": self.confidence_threshold,
            "is_demo": self.is_demo,
            "match_mode": getattr(self, "match_mode", "demo"),
            "task": getattr(self, "task", "demo"),
            "architecture": architecture,
            "architecture_label": getattr(
                self, "architecture_label", ARCHITECTURE_LABELS.get(architecture, architecture)
            ),
        }

    def _predict_softmax(self, image: np.ndarray) -> dict:
        probabilities = self._model_probabilities(image)
        top_indices = np.argsort(probabilities)[::-1][: min(3, len(probabilities))]
        predicted_class = int(top_indices[0])
        confidence = float(probabilities[predicted_class])
        is_recognized = bool(self.class_records) and confidence >= self.confidence_threshold
        winning_label = self._label_for(predicted_class)
        top_3 = [
            {
                "class": int(index),
                "child_id": self._label_for(int(index))["child_id"],
                "name": self._label_for(int(index))["child_name"],
                "confidence": float(probabilities[index]),
                "similarity": None,
            }
            for index in top_indices
        ]
        return {
            **self._base_result(),
            "predicted_class": predicted_class,
            "predicted_id": winning_label["child_id"] if is_recognized else None,
            "predicted_name": winning_label["child_name"] if is_recognized else "Unknown Child",
            "confidence": confidence,
            "confidence_percent": confidence * 100,
            "is_recognized": is_recognized,
            "top_3": top_3,
            "all_probabilities": probabilities.tolist(),
            "similarity": None,
            "enrolled_children": len(self.class_records),
        }

    def _predict_demo(self, image: np.ndarray) -> dict:
        probabilities = self._demo_probabilities(image)
        top_indices = np.argsort(probabilities)[::-1][: min(3, len(probabilities))]
        predicted_class = int(top_indices[0])
        confidence = float(probabilities[predicted_class])
        is_recognized = bool(self.class_records) and confidence >= self.confidence_threshold
        winning_label = self._label_for(predicted_class)
        top_3 = [
            {
                "class": int(index),
                "child_id": self._label_for(int(index))["child_id"],
                "name": self._label_for(int(index))["child_name"],
                "confidence": float(probabilities[index]),
                "similarity": None,
            }
            for index in top_indices
        ]
        return {
            **self._base_result(),
            "predicted_class": predicted_class,
            "predicted_id": winning_label["child_id"] if is_recognized else None,
            "predicted_name": winning_label["child_name"] if is_recognized else "Unknown Child",
            "confidence": confidence,
            "confidence_percent": confidence * 100,
            "is_recognized": is_recognized,
            "top_3": top_3,
            "all_probabilities": probabilities.tolist(),
            "similarity": None,
            "enrolled_children": len(self.class_records),
        }

    def _predict_embedding(self, image: np.ndarray) -> dict:
        self.refresh_enrollment_index()
        if not self._child_embeddings:
            return {
                **self._base_result(),
                "predicted_class": None,
                "predicted_id": None,
                "predicted_name": "Unknown Child",
                "confidence": 0.0,
                "confidence_percent": 0.0,
                "is_recognized": False,
                "top_3": [],
                "all_probabilities": [],
                "similarity": None,
                "enrolled_children": 0,
                "message": "No children are enrolled yet. Enroll a child before identifying.",
            }

        query = self._query_embedding(image)
        scored = []
        for child_id, embedding in self._child_embeddings.items():
            similarity = self._cosine_similarity(query, embedding)
            scored.append((child_id, similarity, self._similarity_confidence(similarity)))
        scored.sort(key=lambda item: item[1], reverse=True)

        child_id, similarity, confidence = scored[0]
        label = self._child_records[child_id]
        is_recognized = confidence >= self.confidence_threshold
        top_3 = [
            {
                "class": index,
                "child_id": candidate_id,
                "name": self._child_records[candidate_id]["child_name"],
                "confidence": candidate_confidence,
                "similarity": candidate_similarity,
            }
            for index, (candidate_id, candidate_similarity, candidate_confidence) in enumerate(scored[:3])
        ]
        return {
            **self._base_result(),
            "predicted_class": top_3[0]["class"] if top_3 else None,
            "predicted_id": label["child_id"] if is_recognized else None,
            "predicted_name": label["child_name"] if is_recognized else "Unknown Child",
            "confidence": confidence,
            "confidence_percent": confidence * 100,
            "is_recognized": is_recognized,
            "top_3": top_3,
            "all_probabilities": [item[2] for item in scored],
            "similarity": similarity,
            "enrolled_children": len(scored),
            "calibration": dict(self.calibration),
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_model_info(self) -> dict:
        """Expose non-sensitive model state for the interface and diagnostics."""
        class_count = len(self.class_names)
        if self.model is not None and not class_count:
            try:
                class_count = int(self.model.output_shape[-1])
            except (AttributeError, IndexError, TypeError):
                class_count = len(self.class_records)
        return {
            "input_shape": (224, 224, 3),
            "num_classes": class_count or len(self.class_records),
            "last_conv_layer": self.last_conv_layer_name,
            "is_demo": self.is_demo,
            "threshold": self.confidence_threshold,
            "architecture": self.architecture,
            "architecture_label": self.architecture_label,
            "task": self.task,
            "match_mode": self.match_mode,
            "class_names": self.class_names,
            "enrolled_children": len(self._child_embeddings) if self.match_mode == "embedding" else len(self.class_records),
            "calibration": dict(self.calibration),
            "model_path": self.model_path,
            "metrics": self.model_metrics,
            "has_registry": self.registry is not None,
        }

    def reload(self) -> None:
        """Reload registry/model state after a new training run."""
        self._enrollment_cache_key = None
        self.registry = None
        self.model = None
        self.embedding_model = None
        self.class_records = {}
        self.class_names = []
        if not self._load_registered_model():
            self._load_legacy_model()
        if self.model is None:
            self._create_demo_model()
