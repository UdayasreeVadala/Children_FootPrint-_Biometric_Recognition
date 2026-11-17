"""Rename enrolled children without retraining.

A child has two labels in this project:

* ``child_id`` (``CHILD_001``) — fixed forever, it is the model's class label
  and the image folder name.
* ``child_name`` (currently ``B001``) — the display name shown in the app.

This script changes **only** ``child_name`` in ``data/enrolled/metadata.json``.
No image is touched and the trained model does not need retraining.

Usage
-----
Show every enrolled child with their current name::

    python rename_children.py --list

Rename a few children inline (key = CHILD id or current name)::

    python rename_children.py --mapping "B001=Aarav,B002=Meera,CHILD_003=Sresta"

Rename from a CSV/TXT file, one ``key,New Name`` per line::

    python rename_children.py --file child_names.csv

Preview without writing anything::

    python rename_children.py --file child_names.csv --dry-run

The file may use ``,``, ``=`` or a tab as the separator, may contain blank
lines, ``#`` comments, and an optional ``key,name`` header row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
METADATA_PATH = BASE_DIR / "data" / "enrolled" / "metadata.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rename enrolled children (display names only)"
    )
    parser.add_argument(
        "--mapping",
        default="",
        help="Comma-separated pairs, e.g. B001=Aarav,B002=Meera",
    )
    parser.add_argument(
        "--file",
        default="",
        help="CSV/TXT file with 'key,New Name' lines",
    )
    parser.add_argument(
        "--metadata",
        default=str(METADATA_PATH),
        help="Path to metadata.json (default: data/enrolled/metadata.json)",
    )
    parser.add_argument("--list", action="store_true", help="List children and exit")
    parser.add_argument("--dry-run", action="store_true", help="Show changes only")
    return parser


def _load_metadata(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Enrollment metadata not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_metadata(path: Path, metadata: dict) -> None:
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    temporary_path.replace(path)


def _parse_mapping(raw: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in filter(None, (piece.strip() for piece in raw.split(","))):
        if "=" not in item:
            raise SystemExit(
                f"Bad --mapping entry {item!r}; expected KEY=New Name"
            )
        key, name = (piece.strip() for piece in item.split("=", 1))
        if key and name:
            pairs.append((key, name))
    return pairs


def _parse_file(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        separator = next(
            (candidate for candidate in (",", "\t", "=") if candidate in stripped),
            None,
        )
        if separator is None:
            raise SystemExit(f"{path.name}:{number}: expected KEY,New Name")
        key, name = (piece.strip() for piece in stripped.split(separator, 1))
        if key.lower() in {"key", "id", "child_id", "child", "subject"}:
            continue  # header row
        if not key or not name:
            raise SystemExit(f"{path.name}:{number}: empty key or name")
        pairs.append((key, name))
    return pairs


def _find_child(metadata: dict, key: str) -> dict | None:
    """Match a key against the child id first, then the current display name."""
    children = metadata.get("children", [])
    for child in children:
        if str(child.get("child_id", "")).lower() == key.lower():
            return child
    matches = [
        child
        for child in children
        if str(child.get("child_name", "")).lower() == key.lower()
    ]
    if len(matches) > 1:
        ids = ", ".join(str(child.get("child_id")) for child in matches)
        raise SystemExit(f"{key!r} matches several children ({ids}); use the CHILD id")
    return matches[0] if matches else None


def main() -> int:
    args = build_parser().parse_args()
    metadata_path = Path(args.metadata)
    metadata = _load_metadata(metadata_path)
    children = metadata.get("children", [])

    if args.list or (not args.mapping and not args.file):
        print(f"{'child_id':12s} {'current name':24s} {'images':>6s}")
        print("-" * 46)
        for child in children:
            print(
                f"{str(child.get('child_id', '')):12s} "
                f"{str(child.get('child_name', ''))[:24]:24s} "
                f"{int(child.get('image_count', 0)):6d}"
            )
        if not args.list:
            print("\nPass --mapping or --file to rename.")
        return 0

    pairs = _parse_mapping(args.mapping)
    if args.file:
        pairs.extend(_parse_file(Path(args.file)))
    if not pairs:
        raise SystemExit("No renames given")

    changes: list[tuple[dict, str, str]] = []
    problems: list[str] = []
    for key, name in pairs:
        name = name.strip()
        if not name:
            problems.append(f"{key}: empty name")
            continue
        if len(name) > 100:
            problems.append(f"{key}: name longer than 100 characters")
            continue
        child = _find_child(metadata, key)
        if child is None:
            problems.append(f"{key}: no enrolled child with that id or name")
            continue
        changes.append((child, key, name))

    if problems:
        print("Nothing was changed. Fix these entries first:\n")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"{'child_id':12s} {'old name':20s} -> {'new name':20s}")
    print("-" * 58)
    changed = 0
    for child, key, name in changes:
        old_name = str(child.get("child_name", ""))
        marker = " (unchanged)" if old_name == name else ""
        print(
            f"{str(child.get('child_id', '')):12s} {old_name[:20]:20s} -> {name[:20]:20s}"
            f"{marker}"
        )
        if old_name != name:
            child["child_name"] = name
            changed += 1

    if args.dry_run:
        print(f"\nDry run: {changed} name(s) would change. Nothing written.")
        return 0

    if not changed:
        print("\nNo names changed.")
        return 0

    _write_metadata(metadata_path, metadata)
    print(f"\nRenamed {changed} child(ren) in {metadata_path}")
    print("No retraining needed: ids and images are unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
