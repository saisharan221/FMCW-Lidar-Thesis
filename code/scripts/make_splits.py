"""Generate 70/15/15 train/val/test splits stratified by scene type.

Spec §3.7: by-sequence split avoids the leakage that happens with
frame-level random splits (frames within a sequence are 100 ms apart
and correlated). Stratification balances city/highway x day/night
across the splits.

Output: configs/splits.json with keys 'train', 'val', 'test', each a
sorted list of sequence UUIDs.

Scene tags come from `data_exploration/full_dataset/stats/per_sequence.json`,
which already labels each sequence with `road_type` (city / highway)
and `lighting` (day / night). If the EDA stats are missing the
script falls back to a deterministic ego-speed + reflectivity
heuristic, but the EDA path is the canonical one.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from ptv3_fmcw.data.aevascenes_dataset import list_sequences


_PER_SEQ_TAGS_PATH = Path("data_exploration/full_dataset/stats/per_sequence.json")
_PER_SEQ_TAGS_CACHE: dict[str, str] | None = None


def _load_eda_tags() -> dict[str, str] | None:
    """Authoritative road/lighting per sequence from the EDA stats file."""
    global _PER_SEQ_TAGS_CACHE
    if _PER_SEQ_TAGS_CACHE is not None:
        return _PER_SEQ_TAGS_CACHE
    if not _PER_SEQ_TAGS_PATH.exists():
        return None
    rows = json.loads(_PER_SEQ_TAGS_PATH.read_text())
    tags: dict[str, str] = {}
    for r in rows:
        uuid = r.get("uuid")
        road = r.get("road_type")
        light = r.get("lighting")
        if uuid and road and light:
            tags[uuid] = f"{road}/{light}"
    _PER_SEQ_TAGS_CACHE = tags or None
    return _PER_SEQ_TAGS_CACHE


def _scene_tag_from_metadata(meta: dict) -> str | None:
    """Read scene tag from EDA stats (canonical) or sequence metadata."""
    eda = _load_eda_tags()
    if eda is not None:
        tag = eda.get(meta["metadata"]["sequence_uuid"])
        if tag:
            return tag
    md = meta.get("metadata", {})
    for key in ("scene_type", "scene_tag", "tags"):
        v = md.get(key)
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            road = v.get("road") or v.get("environment")
            light = v.get("lighting") or v.get("time_of_day")
            if road and light:
                return f"{road}/{light}"
    return None


def _scene_tag_heuristic(meta: dict, root: Path) -> str:
    frames = meta["frames"]
    speeds = []
    for f in frames:
        t = f["ego_pose"]["translation"]
        speeds.append((t["x"], t["y"]))
    arr = np.array(speeds, dtype=np.float64)
    if len(arr) > 1:
        steps = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        mean_speed = float(steps.mean()) / 0.1  # 10 Hz
    else:
        mean_speed = 0.0
    road = "highway" if mean_speed >= 15.0 else "city"

    # Lighting heuristic: mean reflectivity from a single frame's
    # front_wide_lidar npz. Day frames tend to have lower mean retro
    # reflectivity than night (street lights / signs dominate).
    seq_uuid = meta["metadata"]["sequence_uuid"]
    pc_dir = root / seq_uuid / "pointcloud_compensated"
    npz_files = sorted(pc_dir.glob("front_wide_lidar_*.npz"))
    if not npz_files:
        return f"{road}/day"
    npz = np.load(npz_files[len(npz_files) // 2], allow_pickle=True)
    mean_refl = float(npz["reflectivity"].mean())
    light = "night" if mean_refl > 200.0 else "day"
    return f"{road}/{light}"


def scene_tag(root: Path, sequence_uuid: str) -> str:
    meta = json.loads((root / sequence_uuid / "sequence.json").read_text())
    tag = _scene_tag_from_metadata(meta)
    if tag is None:
        tag = _scene_tag_heuristic(meta, root)
    return tag


def stratified_split(
    items_by_tag: dict[str, list[str]],
    fractions: tuple[float, float, float],
    seed: int = 42,
) -> dict[str, list[str]]:
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for tag, items in items_by_tag.items():
        items = sorted(items)
        rng.shuffle(items)
        n = len(items)
        n_train = math.floor(fractions[0] * n)
        n_val = math.floor(fractions[1] * n)
        # Whatever rounds short gets pushed to the test split.
        train = items[:n_train]
        val = items[n_train : n_train + n_val]
        test = items[n_train + n_val :]
        splits["train"].extend(train)
        splits["val"].extend(val)
        splits["test"].extend(test)
    for k in splits:
        splits[k] = sorted(splits[k])
    return splits


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate stratified by-sequence splits.")
    p.add_argument("--data-root", default="data/aevascenes_v0.2", type=Path)
    p.add_argument("--output", default="code/configs/splits.json", type=Path)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument(
        "--fractions",
        nargs=3,
        type=float,
        default=(0.70, 0.15, 0.15),
        help="train / val / test fractions",
    )
    args = p.parse_args(argv)

    seqs = list_sequences(args.data_root)
    if not seqs:
        raise SystemExit(f"No sequences under {args.data_root}")

    by_tag: dict[str, list[str]] = defaultdict(list)
    for s in seqs:
        tag = scene_tag(args.data_root, s)
        by_tag[tag].append(s)

    print(f"[make_splits] {len(seqs)} sequences; tag distribution:")
    for tag in sorted(by_tag):
        print(f"  {tag:>16s}  {len(by_tag[tag])}")

    splits = stratified_split(by_tag, tuple(args.fractions), seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(splits, indent=2))
    print(
        f"[make_splits] wrote {args.output}: "
        f"train={len(splits['train'])} val={len(splits['val'])} "
        f"test={len(splits['test'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
