"""Command-line trainer for FootprintID.

Examples
--------
Train all three architectures on the bundled dataset::

    python train.py

Train only the custom CNN for a quick check::

    python train.py --architectures custom_cnn --epochs 3 --fine-tune-epochs 0

The script preprocesses ``Training_Data`` into ``data/cache/dataset_224`` (the
cache is reused on later runs) and writes models, metrics, confusion matrices,
training curves and ``models/registry.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FootprintID models on a dataset")
    parser.add_argument("--dataset", default=str(BASE_DIR / "Training_Data"))
    parser.add_argument("--models-dir", default=str(BASE_DIR / "models"))
    parser.add_argument("--cache-dir", default=str(BASE_DIR / "data" / "cache" / "dataset_224"))
    parser.add_argument(
        "--architectures",
        default="mobilenetv2,efficientnetb0,custom_cnn",
        help="Comma-separated architectures: mobilenetv2, efficientnetb0, custom_cnn",
    )
    parser.add_argument(
        "--task",
        default="auto",
        choices=["auto", "identity", "side", "visit", "sensor"],
    )
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--fine-tune-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--include-phone", action="store_true")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from app.inference.training import train_dataset_models

    sensors = ["Paper", "Scanner", "Camera"]
    if args.include_phone:
        sensors.append("Phone")

    def progress(event: dict) -> None:
        if args.quiet:
            return
        stage = event.get("stage", "")
        if stage == "preprocessing":
            print(
                f"\r[preprocess] {event.get('current', 0)}/{event.get('total', 0)} "
                f"{event.get('detail', '')[:60]}",
                end="",
                flush=True,
            )
        elif stage == "discovered":
            dataset = event.get("dataset", {})
            print(
                f"\nDiscovered {dataset.get('total_images', 0)} images, "
                f"task={event.get('task')}, subjects={dataset.get('subjects')}"
            )
        elif stage == "split":
            print(f"Class names: {event.get('class_names')}")
            print(f"Split: {json.dumps(event.get('split', {}))}")
        elif stage == "training":
            print(
                f"\n=== Training {event.get('architecture_label')} "
                f"({event.get('index', 0) + 1}/{event.get('total', 0)}) ==="
            )
        elif stage == "architecture_complete":
            metrics = event.get("metrics", {})
            print(
                f" -> accuracy={metrics.get('test_accuracy')} "
                f"top3={metrics.get('test_top3_accuracy')} "
                f"f1={metrics.get('test_macro_f1')} "
                f"time={metrics.get('training_seconds')}s"
            )

    result = train_dataset_models(
        dataset_root=args.dataset,
        models_dir=args.models_dir,
        cache_dir=args.cache_dir,
        task=args.task,
        architectures=[name.strip() for name in args.architectures.split(",") if name.strip()],
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        force_preprocess=args.force_preprocess,
        max_images=args.max_images,
        include_sensors=sensors,
        verbose=0 if args.quiet else 1,
        progress=progress,
    )
    print("\nTraining complete")
    print(f"Best model: {result['best_model']}")
    print(f"Registry:   {result['registry_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
