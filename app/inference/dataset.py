"""Dataset discovery, preprocessing cache, and grouped train/validation splits.

The training data in this project is organised as::

    Training_Data/
        B003/                       # child / subject
            Visit 1/
                Camera/Left/*.JPG
                Paper/Left/*.tif
                Scanner/Left/*.bmp
            Visit 5/
                Left/*.jpeg         # phone camera images

``discover_dataset`` understands that layout, but also accepts a simpler
``dataset/<child_id>/*.jpg`` layout so a larger multi-child dataset can be
dropped in without changing the application.

This module deliberately has no TensorFlow import.  It only uses OpenCV and
NumPy so dataset inspection and caching work even before the ML stack loads.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from app.preprocessing.enhancement import enhance_footprint
from app.preprocessing.roi_detection import process_footprint

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SENSOR_NAMES = {"camera", "paper", "scanner", "phone", "video"}
TASK_NAMES = ("identity", "side", "visit", "sensor")

_VISIT_RE = re.compile(r"(?:visit|v)[\s_-]*(\d+)", re.IGNORECASE)
_SIDE_RE = re.compile(r"(?:^|[_\-\s])(left|right|l|r)(?:$|[_\-\s])", re.IGNORECASE)


@dataclass(frozen=True)
class DatasetSample:
    """A single source image and the metadata parsed from its path."""

    source_path: str
    relative_path: str
    subject: str
    visit: str
    side: str
    sensor: str
    group: str

    def label_for(self, task: str) -> str:
        task = resolve_task_name(task)
        if task == "identity":
            return self.subject
        if task == "side":
            return self.side
        if task == "visit":
            return self.visit
        if task == "sensor":
            return self.sensor
        raise ValueError(f"Unknown task: {task}")


@dataclass
class PreprocessedSample:
    """A cached, 224×224 enhanced image ready for model training."""

    cache_path: str
    source_path: str
    relative_path: str
    subject: str
    visit: str
    side: str
    sensor: str
    group: str
    roi_ok: bool
    roi_area_ratio: float
    width: int
    height: int
    label: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def label_for(self, task: str) -> str:
        task = resolve_task_name(task)
        if task == "identity":
            return self.subject
        if task == "side":
            return self.side
        if task == "visit":
            return self.visit
        if task == "sensor":
            return self.sensor
        raise ValueError(f"Unknown task: {task}")


def resolve_task_name(task: str) -> str:
    normalized = str(task or "auto").strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized not in TASK_NAMES:
        raise ValueError(
            f"Unknown task '{task}'. Choose from: auto, " + ", ".join(TASK_NAMES)
        )
    return normalized


def _clean_segment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return cleaned or "unknown"


def _guess_side(parts: Iterable[str]) -> str:
    for part in parts:
        lowered = part.strip().lower()
        if lowered in {"left", "l"}:
            return "Left"
        if lowered in {"right", "r"}:
            return "Right"
    # Fall back to filename markers such as BP_003_V1_L_1.tif.
    joined = "_".join(parts)
    match = _SIDE_RE.search(joined)
    if match:
        return "Left" if match.group(1).lower().startswith("l") else "Right"
    return "Unknown"


def _guess_visit(parts: Iterable[str]) -> str:
    for part in parts:
        match = _VISIT_RE.search(part)
        if match:
            return f"Visit {int(match.group(1))}"
    return "Visit 1"


def _guess_sensor(parts: Iterable[str]) -> str:
    for part in parts:
        lowered = part.strip().lower()
        if lowered in SENSOR_NAMES:
            return lowered.capitalize()
    return "Phone"


def discover_dataset(
    root: str | Path,
    include_sensors: Iterable[str] | None = None,
) -> list[DatasetSample]:
    """Walk *root* and return every supported image with parsed metadata.

    ``include_sensors`` limits the result to selected acquisition modalities
    (for example ``{"Paper", "Scanner"}``).  Phone photographs are excluded by
    default in the training script because their cluttered background makes
    the plantar ROI unreliable; clean scans are the intended biometric input.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise ValueError(f"Dataset directory does not exist: {root_path}")
    if not root_path.is_dir():
        raise ValueError(f"Dataset path is not a directory: {root_path}")

    allowed_sensors = (
        {str(sensor).strip().lower() for sensor in include_sensors}
        if include_sensors
        else None
    )

    samples: list[DatasetSample] = []
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.name.startswith("."):
            continue

        relative = path.relative_to(root_path)
        parts = list(relative.parts)
        if len(parts) < 2:
            # A loose image at the dataset root cannot be assigned an identity.
            continue

        subject = parts[0]
        visit = _guess_visit(parts[:-1])
        # Include the filename: imported datasets commonly encode L/R in the
        # filename rather than in a directory (for example ``scan-left.png``).
        side = _guess_side([*parts[:-1], path.stem])
        sensor = _guess_sensor(parts[:-1])
        if allowed_sensors is not None and sensor.lower() not in allowed_sensors:
            continue
        # A capture group identifies one acquisition session, not a class
        # label.  Keeping both halves of a split adult scan in this one group
        # prevents a left/right pair from the same scan being split between
        # training and evaluation.
        group = "|".join(
            [_clean_segment(subject), _clean_segment(visit), _clean_segment(sensor)]
        )
        samples.append(
            DatasetSample(
                source_path=str(path.resolve()),
                relative_path=str(relative),
                subject=subject,
                visit=visit,
                side=side,
                sensor=sensor,
                group=group,
            )
        )
    return samples


def summarize_samples(samples: list[DatasetSample]) -> dict:
    """Return counts used by the UI and training reports."""
    subjects = sorted({sample.subject for sample in samples})
    sides = sorted({sample.side for sample in samples if sample.side != "Unknown"})
    visits = sorted(
        {sample.visit for sample in samples},
        key=lambda value: int(re.search(r"\d+", value).group()) if re.search(r"\d+", value) else 0,
    )
    sensors = sorted({sample.sensor for sample in samples})

    by_visit: dict[str, dict[str, int]] = {}
    for sample in samples:
        bucket = by_visit.setdefault(sample.visit, {"total": 0, "left": 0, "right": 0})
        bucket["total"] += 1
        if sample.side == "Left":
            bucket["left"] += 1
        elif sample.side == "Right":
            bucket["right"] += 1

    return {
        "total_images": len(samples),
        "subjects": subjects,
        "subject_count": len(subjects),
        "sides": sides,
        "visits": visits,
        "sensors": sensors,
        "by_visit": by_visit,
    }


def resolve_task(samples: list[DatasetSample], requested: str = "auto") -> str:
    """Choose a meaningful label task, even when only one child is present."""
    requested = resolve_task_name(requested)
    if requested != "auto":
        return requested

    subjects = {sample.subject for sample in samples}
    sides = {sample.side for sample in samples if sample.side != "Unknown"}
    visits = {sample.visit for sample in samples}
    sensors = {sample.sensor for sample in samples}

    # A real child-recognition dataset needs at least two identities.  With a
    # single subject we train a left/right classifier, whose penultimate
    # embedding is then used for child verification in the web application.
    if len(subjects) >= 2:
        return "identity"
    if len(sides) >= 2:
        return "side"
    if len(visits) >= 2:
        return "visit"
    if len(sensors) >= 2:
        return "sensor"
    return "identity"


def _read_image(path: Path) -> np.ndarray:
    """Read an image, normalising 16-bit TIFF/PNG and alpha channels to BGR."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")

    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        minimum, maximum = float(image.min()), float(image.max())
        if maximum > minimum:
            image = (image - minimum) * (255.0 / (maximum - minimum))
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Unsupported image shape {image.shape}: {path}")
    return image


def _resize_for_processing(image: np.ndarray, max_side: int = 1600) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _roi_is_usable(roi: np.ndarray, area_ratio: float) -> bool:
    """Reject almost-black or almost-empty crops from failed segmentation."""
    if roi is None or roi.size == 0:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    if float(gray.std()) < 6.0:
        return False
    non_dark = float(np.mean(gray > 12))
    if non_dark < 0.10:
        return False
    # A contour that covers nearly the entire frame is suspicious for a photo
    # with a cluttered background, but still allow it as a last resort.
    if area_ratio > 0.97:
        return False
    return True


def _fallback_enhance(image: np.ndarray) -> tuple[np.ndarray, bool, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
    enhanced = enhance_footprint(resized)["enhanced"]
    return enhanced, False, 1.0


def preprocess_sample(source_path: str | Path) -> tuple[np.ndarray, bool, float]:
    """Run ROI detection + Gabor enhancement, with a safe full-frame fallback."""
    path = Path(source_path)
    image = _read_image(path)
    image = _resize_for_processing(image)

    try:
        result = process_footprint(image, return_visualization=False)
        roi = result["roi"]
        region_info = result.get("region_info", {})
        bbox = region_info.get("bounding_box")
        area_ratio = 1.0
        if bbox:
            _, _, width, height = bbox
            area_ratio = float(width * height) / float(image.shape[0] * image.shape[1])
        if _roi_is_usable(roi, area_ratio):
            enhanced = enhance_footprint(roi)["enhanced"]
            return enhanced, True, area_ratio
    except Exception:
        pass

    return _fallback_enhance(image)


def preprocess_dataset(
    samples: list[DatasetSample],
    cache_dir: str | Path,
    *,
    force: bool = False,
    progress: Callable[[int, int, object], None] | None = None,
) -> list[PreprocessedSample]:
    """Preprocess every sample once and cache the 224×224 enhanced PNGs."""
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest: list[PreprocessedSample] = []

    for index, sample in enumerate(samples):
        source = Path(sample.source_path)
        destination = (
            cache_root
            / _clean_segment(sample.subject)
            / _clean_segment(sample.visit)
            / _clean_segment(sample.sensor)
            / _clean_segment(sample.side)
            / f"{_clean_segment(source.stem)}.png"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists() and not force and destination.stat().st_mtime >= source.stat().st_mtime:
            # Reconstruct metadata without recomputing the image.
            manifest.append(
                PreprocessedSample(
                    cache_path=str(destination.resolve()),
                    source_path=str(source.resolve()),
                    relative_path=sample.relative_path,
                    subject=sample.subject,
                    visit=sample.visit,
                    side=sample.side,
                    sensor=sample.sensor,
                    group=sample.group,
                    roi_ok=True,
                    roi_area_ratio=0.0,
                    width=224,
                    height=224,
                )
            )
            if progress:
                progress(index + 1, len(samples), sample)
            continue

        try:
            enhanced, roi_ok, roi_area_ratio = preprocess_sample(source)
            if not cv2.imwrite(str(destination), enhanced):
                raise RuntimeError("cv2.imwrite failed")
            manifest.append(
                PreprocessedSample(
                    cache_path=str(destination.resolve()),
                    source_path=str(source.resolve()),
                    relative_path=sample.relative_path,
                    subject=sample.subject,
                    visit=sample.visit,
                    side=sample.side,
                    sensor=sample.sensor,
                    group=sample.group,
                    roi_ok=roi_ok,
                    roi_area_ratio=round(float(roi_area_ratio), 4),
                    width=int(enhanced.shape[1]),
                    height=int(enhanced.shape[0]),
                )
            )
        except Exception as error:
            # A single unreadable file must not abort a long preprocessing run.
            if progress:
                progress(index + 1, len(samples), f"{sample.relative_path} (skipped: {error})")
            continue

        if progress:
            progress(index + 1, len(samples), sample.relative_path)

    return manifest


def save_manifest(manifest: list[PreprocessedSample], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump([sample.to_dict() for sample in manifest], file, indent=2)
    temporary.replace(destination)
    return destination


def load_manifest(path: str | Path) -> list[PreprocessedSample]:
    source = Path(path)
    if not source.exists():
        raise ValueError(f"Dataset manifest not found: {source}")
    with source.open("r", encoding="utf-8") as file:
        records = json.load(file)
    return [PreprocessedSample(**record) for record in records]


def _stable_seed(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def grouped_split(
    samples: list[PreprocessedSample],
    task: str,
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[PreprocessedSample], list[PreprocessedSample], list[PreprocessedSample]]:
    """Split by capture group so near-duplicate images cannot leak across sets."""
    if not samples:
        raise ValueError("No preprocessed samples to split")
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Validation and test fractions must be positive")
    if validation_fraction + test_fraction >= 0.9:
        raise ValueError("Validation + test fractions must leave a training set")

    labels = sorted({sample.label_for(task) for sample in samples})
    train: list[PreprocessedSample] = []
    validation: list[PreprocessedSample] = []
    test: list[PreprocessedSample] = []

    for label in labels:
        label_samples = [sample for sample in samples if sample.label_for(task) == label]
        groups: dict[str, list[PreprocessedSample]] = {}
        for sample in label_samples:
            groups.setdefault(sample.group, []).append(sample)

        ordered_groups = sorted(
            groups.items(),
            key=lambda item: _stable_seed(item[0], seed),
        )
        group_count = len(ordered_groups)
        if group_count < 3:
            # Too few groups: put them all in training rather than making a
            # misleading split with one near-duplicate group in test.
            train.extend(sample for _, group in ordered_groups for sample in group)
            continue

        # Allocate whole capture groups while targeting a fraction of the
        # *images* (not the number of groups).  Group sizes vary a lot between
        # sensor and visit, so counting groups alone can create a badly
        # imbalanced test set.
        total_images = len(label_samples)
        test_target = total_images * test_fraction
        validation_target = total_images * validation_fraction
        remaining = list(ordered_groups)

        test_groups: list[tuple[str, list[PreprocessedSample]]] = []
        accumulated = 0
        while remaining and len(test_groups) < max(1, group_count - 2):
            candidate = remaining.pop(0)
            test_groups.append(candidate)
            accumulated += len(candidate[1])
            if accumulated >= test_target:
                break

        validation_groups: list[tuple[str, list[PreprocessedSample]]] = []
        accumulated = 0
        max_validation = max(1, group_count - len(test_groups) - 1)
        while remaining and len(validation_groups) < max_validation:
            candidate = remaining.pop(0)
            validation_groups.append(candidate)
            accumulated += len(candidate[1])
            if accumulated >= validation_target:
                break

        train_groups = remaining

        test.extend(sample for _, group in test_groups for sample in group)
        validation.extend(sample for _, group in validation_groups for sample in group)
        train.extend(sample for _, group in train_groups for sample in group)

    if not train:
        raise ValueError("The grouped split produced an empty training set")
    if not validation:
        # Fall back to a stratified image-level validation split.
        train, validation = _image_level_split(train or samples, task, validation_fraction, seed)
    if not test:
        test = list(validation)

    return _shuffle(train, seed), _shuffle(validation, seed + 1), _shuffle(test, seed + 2)


def _shuffle(samples: list[PreprocessedSample], seed: int) -> list[PreprocessedSample]:
    return sorted(samples, key=lambda sample: _stable_seed(sample.cache_path, seed))


def _image_level_split(
    samples: list[PreprocessedSample],
    task: str,
    validation_fraction: float,
    seed: int,
) -> tuple[list[PreprocessedSample], list[PreprocessedSample]]:
    labels = sorted({sample.label_for(task) for sample in samples})
    train: list[PreprocessedSample] = []
    validation: list[PreprocessedSample] = []
    for label in labels:
        label_samples = _shuffle(
            [sample for sample in samples if sample.label_for(task) == label],
            seed + _stable_seed(label, seed),
        )
        count = max(1, round(len(label_samples) * validation_fraction))
        validation.extend(label_samples[:count])
        train.extend(label_samples[count:])
    return train, validation


def class_distribution(samples: list[PreprocessedSample], task: str) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for sample in samples:
        label = sample.label_for(task)
        distribution[label] = distribution.get(label, 0) + 1
    return dict(sorted(distribution.items()))
