#!/usr/bin/env python3
"""Apply the IIoU crop+resize recipe to a directory of user-downloaded raws.

Release workflow (user side):
  1. For each entry in test_iiou_urls.json, download the URL (right column)
     and save it at the path given by the key (left column), e.g.
     `datasets/test_iiou_raw/Fruits/Apple/iiou_001.jpg`. The key's extension
     matches what the URL serves — save with that exact name.
  2. Run this script:
         python datasets/preprocess_iiou.py
  3. The 224x224, aligned outputs land at
     `datasets/test_iiou/<Family>/<Ingredient>/iiou_NNN.jpg`
     so they line up with `mask_iiou_first/` and `mask_iiou_second/`.

Only depends on Pillow.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

TARGET = 224
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR.parent / "datasets"

RAW_PREFIX = "test_iiou_raw/"


def raw_key_to_out_rel(raw_key: str) -> str:
    """`test_iiou_raw/Family/Ingredient/iiou_NNN.<ext>` → `Family/Ingredient/iiou_NNN.jpg`."""
    if not raw_key.startswith(RAW_PREFIX):
        raise ValueError(f"recipe key must start with {RAW_PREFIX!r}: {raw_key!r}")
    rel = raw_key[len(RAW_PREFIX):]
    return Path(rel).with_suffix(".jpg").as_posix()


def process(raw_key: str, recipe: dict, raw_root: Path, out_root: Path, force: bool) -> str:
    rel_raw = raw_key[len(RAW_PREFIX):]
    raw = raw_root / rel_raw
    out = out_root / raw_key_to_out_rel(raw_key)

    if out.is_file() and not force:
        return "skipped"
    if not raw.is_file():
        return "raw_missing"

    img = Image.open(raw).convert("RGB")
    status = recipe.get("status")
    if status == "crop_resize":
        box = recipe.get("crop_box")
        if not box or len(box) != 4:
            return "bad_recipe"
        x, y, w, h = box
        img = img.crop((x, y, x + w, y + h))
    elif status != "resize_only":
        return f"unknown_status:{status}"
    img = img.resize((TARGET, TARGET), Image.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "JPEG", quality=95)
    return "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA),
                    help="Directory holding test_iiou_recipe.json (default: ../datasets).")
    ap.add_argument("--raw-dir", default=None,
                    help="Where the user's raw downloads live (default: <data-dir>/test_iiou_raw).")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write processed 224x224 images (default: <data-dir>/test_iiou).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing outputs.")
    args = ap.parse_args()

    data = Path(args.data_dir).resolve()
    raw = Path(args.raw_dir).resolve() if args.raw_dir else data / "test_iiou_raw"
    out = Path(args.out_dir).resolve() if args.out_dir else data / "test_iiou"
    recipe_file = data / "test_iiou_recipe.json"
    if not recipe_file.is_file():
        sys.exit(f"[FATAL] recipe not found: {recipe_file}")
    recipes = json.load(open(recipe_file))
    print(f"Processing {len(recipes)} entries:")
    print(f"  raw   : {raw}")
    print(f"  out   : {out}")
    print(f"  force : {args.force}\n")

    counts: Counter = Counter()
    missing = []
    for raw_key, rcp in sorted(recipes.items()):
        status = process(raw_key, rcp, raw, out, args.force)
        counts[status] += 1
        if status == "raw_missing":
            missing.append(raw_key)

    print("Summary:")
    for k in ("ok", "skipped", "raw_missing", "bad_recipe"):
        if counts.get(k):
            print(f"  {k:14s} {counts[k]}")
    other = {k: v for k, v in counts.items() if k not in ("ok", "skipped", "raw_missing", "bad_recipe")}
    for k, v in other.items():
        print(f"  {k:14s} {v}")
    if missing:
        print(f"\nMissing raw files ({len(missing)}). Each recipe key tells you exactly")
        print(f"where to save: datasets/<key> — see test_iiou_urls.json for URLs.")
        for p in missing[:20]:
            print(f"  - {p}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        sys.exit(1)


if __name__ == "__main__":
    main()
