"""Bulk-enrol subject folders into ``data/enrolled``.

Walks a dataset root (default ``Training_Data``), maps each subject folder to a
``CHILD_xxx`` id, and runs the exact same pipeline the web API uses:

    ROI detection -> foot mask -> CLAHE/Gabor -> crease focus

The processed 224x224 PNGs are written to ``data/enrolled/CHILD_xxx/`` and
``metadata.json`` is updated.  Existing enrolled children are **never modified**
unless ``--overwrite`` is passed, so it is safe to run repeatedly as new
subjects arrive.

Examples
--------
Preview the plan without touching anything::

    python bulk_enroll.py --dry-run

Enrol every subject that is not enrolled yet (max 20 images each)::

    python bulk_enroll.py

Enrol selected subjects with a custom count and name::

    python bulk_enroll.py --only B012,B013 --max-images 15

Force a specific mapping::

    python bulk_enroll.py --mapping B012=CHILD_012,B100=CHILD_020

Enrol only sharp images (drops blurry frames)::

    python bulk_enroll.py --min-sharpness 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bulk-enrol subject folders")
    parser.add_argument("--source", default=str(BASE_DIR / "Training_Data"))
    parser.add_argument("--data-dir", default=str(BASE_DIR / "data" / "enrolled"))
    parser.add_argument("--only", default="", help="Comma-separated subject names")
    parser.add_argument("--mapping", default="", help="SUBJECT=CHILD_ID pairs")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--min-sharpness", type=float, default=0.0)
    parser.add_argument("--min-foreground", type=float, default=0.02)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing images in the target child folder first.",
    )
    return parser


def _images_in(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not path.name.startswith(".")
    )


def _subjects(source: Path) -> list[str]:
    return sorted(
        entry.name
        for entry in source.iterdir()
        if entry.is_dir() and _images_in(entry)
    )


def _evenly_spaced(paths: list[Path], limit: int) -> list[Path]:
    if limit <= 0 or len(paths) <= limit:
        return paths
    step = len(paths) / float(limit)
    return [paths[min(len(paths) - 1, int(index * step))] for index in range(limit)]


def _sharpness(path: Path) -> float:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0
    height, width = image.shape[:2]
    scale = 256.0 / max(height, width)
    if scale < 1.0:
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def _load_metadata(path: Path) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    return {"children": []}


def _write_metadata(path: Path, metadata: dict) -> None:
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    temporary_path.replace(path)


def _target_directory(data_dir: Path, child_id: str) -> Path:
    candidate = (data_dir / child_id).resolve()
    if data_dir.resolve() not in candidate.parents:
        raise SystemExit(f"Invalid child id: {child_id}")
    return candidate


def _parse_mapping(raw: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in filter(None, (piece.strip() for piece in raw.split(","))):
        if "=" not in item:
            raise SystemExit(f"Bad --mapping entry: {item!r} (expected SUBJECT=CHILD_ID)")
        subject, child_id = (piece.strip() for piece in item.split("=", 1))
        mapping[subject] = child_id
    return mapping


def _default_child_id(subject: str) -> str | None:
    match = re.fullmatch(r"[Bb](\d+)", subject)
    if match:
        return f"CHILD_{int(match.group(1)):03d}"
    return None


def main() -> int:
    args = build_parser().parse_args()
    from app.preprocessing.enhancement import enhance_footprint, focus_on_creases
    from app.preprocessing.roi_detection import process_footprint

    source = Path(args.source)
    data_dir = Path(args.data_dir)
    if not source.is_dir():
        raise SystemExit(f"Source directory not found: {source}")

    metadata_path = data_dir / "metadata.json"
    metadata = _load_metadata(metadata_path)
    known_ids = {child["child_id"] for child in metadata.get("children", [])}

    subjects = _subjects(source)
    only = {piece.strip() for piece in args.only.split(",") if piece.strip()}
    if only:
        subjects = [subject for subject in subjects if subject in only]

    mapping = _parse_mapping(args.mapping)
    next_index = 1
    while f"CHILD_{next_index:03d}" in known_ids:
        next_index += 1

    plan: list[tuple[str, str, list[Path]]] = []
    for subject in subjects:
        if subject in mapping:
            child_id = mapping[subject]
        else:
            child_id = _default_child_id(subject)
            if child_id is None:
                child_id = f"CHILD_{next_index:03d}"
                next_index += 1
        plan.append((subject, child_id, _images_in(source / subject)))

    print(f"source : {source}")
    print(f"target : {data_dir}")
    print(f"subjects found: {len(plan)}\n")
    print(f"{'subject':10s} {'child_id':12s} {'images':>6s} {'take':>5s} {'state':>12s}")
    print("-" * 54)

    selected: list[tuple[str, str, list[Path]]] = []
    for subject, child_id, paths in plan:
        target = data_dir / child_id
        existing = len(list(target.glob("*.png"))) if target.exists() else 0
        take = min(len(paths), args.max_images) if args.max_images > 0 else len(paths)
        if existing and not args.overwrite:
            state = f"skip ({existing})"
        elif len(paths) < 2:
            state = "skip (<2)"
        else:
            state = "enrol" if not existing else "overwrite"
            selected.append((subject, child_id, paths))
        print(f"{subject:10s} {child_id:12s} {len(paths):6d} {take:5d} {state:>12s}")

    if args.dry_run:
        print(f"\nDry run: {len(selected)} subject(s) would be enrolled. Nothing written.")
        return 0

    if not selected:
        print("\nNothing to enrol.")
        return 0

    print()
    seen_hashes: set[str] = set()
    total_written = 0
    for subject, child_id, paths in selected:
        target = _target_directory(data_dir, child_id)
        if args.overwrite and target.exists():
            for path in target.glob("*.png"):
                path.unlink()
        target.mkdir(parents=True, exist_ok=True)

        candidates = _evenly_spaced(paths, args.max_images)
        if args.min_sharpness > 0:
            sharp = [(path, _sharpness(path)) for path in candidates]
            candidates = [path for path, score in sharp if score >= args.min_sharpness]
            if len(candidates) < 2:
                print(f"{subject}: fewer than 2 sharp images, skipped")
                continue

        written = duplicates = blank = failed = 0
        for path in candidates:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                failed += 1
                continue
            try:
                roi_result = process_footprint(image, return_visualization=True)
                enhancement = enhance_footprint(roi_result["roi"])
                focused = focus_on_creases(
                    enhancement["enhanced"], mask=roi_result.get("roi_mask")
                )
            except Exception:
                failed += 1
                continue

            if focused.ndim == 3:
                focused = cv2.cvtColor(focused, cv2.COLOR_BGR2GRAY)
            if focused.shape[:2] != (224, 224):
                focused = cv2.resize(focused, (224, 224), interpolation=cv2.INTER_AREA)
            if float(np.mean(focused > 0)) < args.min_foreground:
                blank += 1
                continue

            digest = hashlib.sha1(focused.tobytes()).hexdigest()
            if digest in seen_hashes:
                duplicates += 1
                continue

            seen_hashes.add(digest)
            destination = target / f"{child_id}_{digest[:8]}.png"
            if cv2.imwrite(str(destination), focused):
                written += 1
            else:
                failed += 1

        count = len(list(target.glob("*.png")))
        found = False
        for child in metadata.setdefault("children", []):
            if child["child_id"] == child_id:
                child["child_name"] = subject
                child["image_count"] = count
                found = True
                break
        if not found:
            metadata["children"].append(
                {"child_id": child_id, "child_name": subject, "image_count": count}
            )
        total_written += written
        print(
            f"{subject} -> {child_id}: {written} written, {count} total "
            f"(duplicates={duplicates}, blank={blank}, failed={failed})"
        )

    _write_metadata(metadata_path, metadata)
    print(f"\nDone: {total_written} new image(s) across {len(selected)} child(ren).")
    print("Next: retrain once with  POST /api/retrain  (or the UI button).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
