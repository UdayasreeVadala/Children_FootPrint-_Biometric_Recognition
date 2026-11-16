"""Materialise the heavy training augmentation as image files.

Training uses on-the-fly augmentation, so no augmented images normally exist on
disk.  This script runs the exact same pipeline and writes the variants out so
they can be inspected, shared, or fed to external tools.

Examples
--------
Write 10 variants per enrolled image into ``data/augmented``::

    python export_augmented.py

Write 25 variants and a contact sheet for one child::

    python export_augmented.py --copies 25 --preview CHILD_001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export augmented enrollment images")
    parser.add_argument("--data-dir", default=str(BASE_DIR / "data" / "enrolled"))
    parser.add_argument("--output", default=str(BASE_DIR / "data" / "augmented"))
    parser.add_argument("--copies", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preview",
        default="",
        help="Child id to build a contact-sheet preview for (first child if empty).",
    )
    parser.add_argument("--preview-count", type=int, default=8)
    return parser


def write_contact_sheet(child_dir: Path, child_id: str, count: int) -> Path | None:
    paths = sorted(child_dir.glob("*.png"))[:count]
    if not paths:
        return None
    tiles = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        tile = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.putText(
            tile,
            path.stem[-12:],
            (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            200,
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    if not tiles:
        return None
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[index : index + 4]) for index in range(0, len(tiles), 4)]
    sheet = np.vstack(rows)
    path = child_dir / f"{child_id}_preview.png"
    cv2.imwrite(str(path), sheet)
    return path


def main() -> int:
    args = build_parser().parse_args()
    from app.inference.training import export_augmented_dataset

    result = export_augmented_dataset(
        Path(args.data_dir),
        Path(args.output),
        copies_per_image=args.copies,
        seed=args.seed,
    )
    print(f"output        : {result['output_dir']}")
    print(f"children      : {result['children']}")
    print(f"source images : {result['source_images']}")
    print(f"copies/image  : {result['copies_per_image']}")
    print(f"images written: {result['images_written']}")
    for child_id, count in sorted(result["per_child"].items()):
        print(f"  {child_id}: {count}")

    output_dir = Path(result["output_dir"])
    child_ids = sorted(result["per_child"])
    preview_child = args.preview or (child_ids[0] if child_ids else "")
    if preview_child:
        sheet = write_contact_sheet(
            output_dir / preview_child, preview_child, args.preview_count
        )
        if sheet is not None:
            print(f"preview       : {sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
