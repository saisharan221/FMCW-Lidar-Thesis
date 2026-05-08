"""CLI entry point: run B0-B3 on val or test and write the deliverables.

Spec §10:
    python scripts/eval_baselines.py \\
        --config configs/baseline.yaml \\
        --split {val,test} \\
        --baselines B0 B1 B2 B3 \\
        --output frozen_test/<auto-datetime>/

Outputs (spec §10.2):
    LOCKED.txt        (test split only)
    git_commit.txt
    config.yaml       (resolved config)
    predictions/<baseline>.npz
    metrics.json      (per-baseline nested dict)
    sign_check.json   (B1 sign-check banner contents)
    tables/T1_main_results.tex
    tables/T2_per_class_epet.tex
    figures/F3_tangential_dominance.pdf
    figures/F4_per_class_epe.pdf
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from ptv3_fmcw.data.aevascenes_dataset import AevaScenesFrameDataset, list_sequences  # noqa: E402
from ptv3_fmcw.eval.baselines import (  # noqa: E402
    B0_Zero,
    B1_DopplerOnly,
    B2_ClassMean,
    B3_DopplerPlusClassMean,
    fit_class_mean,
)
from ptv3_fmcw.eval.evaluate import (  # noqa: E402
    b1_sign_check,
    evaluate_baselines,
    print_sign_check_banner,
)
from ptv3_fmcw.eval.tables import generate_T1, generate_T2  # noqa: E402
from ptv3_fmcw.eval.visualize import generate_F3, generate_F4  # noqa: E402


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _config_hash(cfg: dict) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = path.read_text()
    # Resolve `${oc.env:VAR, default}` patterns without omegaconf.
    import re

    def _env_sub(m):
        var, default = m.group(1), m.group(2)
        return os.environ.get(var, default).strip()
    raw = re.sub(r"\$\{oc\.env:([A-Z_]+),\s*([^}]+)\}", _env_sub, raw)
    return yaml.safe_load(raw)


def _scene_tag_for(root: Path, splits: dict) -> dict[str, str]:
    """Compute scene tag per sequence using the heuristic in make_splits."""
    from scripts.make_splits import scene_tag  # type: ignore
    tags: dict[str, str] = {}
    for split_seqs in splits.values():
        for s in split_seqs:
            if s not in tags:
                tags[s] = scene_tag(root, s)
    return tags


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="code/configs/baseline.yaml", type=Path)
    p.add_argument("--data-root", default=None, type=Path,
                   help="overrides config's data.root (or AEVASCENES_ROOT)")
    p.add_argument("--splits", default=None, type=Path,
                   help="overrides config's data.splits_file")
    p.add_argument("--split", default="val", choices=("train", "val", "test"))
    p.add_argument("--baselines", nargs="+", default=["B0", "B1", "B2", "B3"])
    p.add_argument("--output", default=None, type=Path)
    p.add_argument("--max-train-frames", default=None, type=int,
                   help="cap on training frames used to fit class-mean")
    p.add_argument("--max-eval-frames", default=None, type=int,
                   help="cap on eval frames (for smoke runs)")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-tables", action="store_true")
    p.add_argument("--no-predictions", action="store_true",
                   help="don't dump full predictions/*.npz (saves disk)")
    args = p.parse_args(argv)

    cfg = _load_config(args.config)
    data_root = args.data_root or Path(cfg["data"]["root"])
    splits_path = args.splits or Path(cfg["data"]["splits_file"])

    if args.split == "test":
        cfg.setdefault("output", {})
        cfg["output"]["frozen_test_split"] = True

    if args.output is None:
        ts = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        root_dir = Path(cfg.get("output", {}).get("frozen_test_root", "frozen_test"))
        args.output = root_dir / ts

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"[eval] data_root  = {data_root}")
    print(f"[eval] splits     = {splits_path}")
    print(f"[eval] eval_split = {args.split}")
    print(f"[eval] output     = {args.output}")

    if not splits_path.exists():
        raise SystemExit(
            f"splits file not found: {splits_path}\n"
            f"run: python code/scripts/make_splits.py --output {splits_path}"
        )
    splits = json.loads(splits_path.read_text())
    if args.split not in splits:
        raise SystemExit(f"split {args.split!r} not in {splits_path}")

    # Scene tag per sequence (cached in dataset).
    scene_tag_map = _scene_tag_for(data_root, splits)
    scene_tag_fn = lambda uuid: scene_tag_map.get(uuid, "unknown")  # noqa: E731

    train_ds = AevaScenesFrameDataset(
        data_root, splits["train"], scene_tag_fn=scene_tag_fn,
    )
    eval_ds = AevaScenesFrameDataset(
        data_root, splits[args.split], scene_tag_fn=scene_tag_fn,
    )
    if args.max_train_frames and len(train_ds) > args.max_train_frames:
        train_ds._index = train_ds._index[: args.max_train_frames]
    if args.max_eval_frames and len(eval_ds) > args.max_eval_frames:
        eval_ds._index = eval_ds._index[: args.max_eval_frames]
    print(f"[eval] train frames = {len(train_ds)}")
    print(f"[eval] eval frames  = {len(eval_ds)}")

    # Sign-check banner (spec §8.2).
    sign_report = b1_sign_check(eval_ds, max_frames=min(50, len(eval_ds)))
    print_sign_check_banner(sign_report)
    (args.output / "sign_check.json").write_text(json.dumps(sign_report, indent=2))

    # Apply sign flip if config requested.
    sign_cfg = cfg.get("output", {}).get("v_radial_sign", "positive_recedes")
    if sign_cfg == "positive_approaches":
        print("[eval] config requests v_radial sign flip — patching baselines.")
        # The sign flip happens at the predict level for B1/B3 (only consumers).
        # We monkey-patch v_radial in the dataset record by wrapping the loader.
        original_load = eval_ds.load
        def _flipped(uuid, fi, _orig=original_load):
            r = _orig(uuid, fi)
            r["v_radial"] = -r["v_radial"]
            r["feat"][:, 4] = -r["feat"][:, 4]
            return r
        eval_ds.load = _flipped  # type: ignore

    # Fit B2 / B3.
    selected = set(args.baselines)
    baselines: dict[str, object] = {}
    if "B0" in selected:
        baselines["B0_zero"] = B0_Zero()
    if "B1" in selected:
        baselines["B1_doppler_only"] = B1_DopplerOnly()
    needs_class_mean = ("B2" in selected) or ("B3" in selected)
    if needs_class_mean:
        print(f"[eval] fitting class means on {len(train_ds)} train frames ...")
        class_means = fit_class_mean(
            (train_ds[i] for i in range(len(train_ds))),
            n_total=len(train_ds),
        )
        print(f"[eval]   {len(class_means)} classes seen")
        if "B2" in selected:
            baselines["B2_class_mean"] = B2_ClassMean(class_means=class_means)
        if "B3" in selected:
            baselines["B3_doppler_plus_class_mean"] = B3_DopplerPlusClassMean(
                class_means=class_means,
            )

    # Run.
    pred_dir = None if args.no_predictions else (args.output / "predictions")
    print(f"[eval] running {len(baselines)} baselines on {len(eval_ds)} frames ...")
    results = evaluate_baselines(baselines, eval_ds, save_predictions_dir=pred_dir)

    # Persist outputs.
    out = {
        "split": args.split,
        "n_frames": len(eval_ds),
        "n_train_frames_for_class_mean": len(train_ds) if needs_class_mean else 0,
        "baselines": results,
        "sign_check": sign_report,
        "metadata": {
            "git_commit": _git_commit(),
            "config_hash": _config_hash(cfg),
            "run_started_at": dt.datetime.now(dt.UTC).isoformat(),
            "config_path": str(args.config),
            "data_root": str(data_root),
            "splits_path": str(splits_path),
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(out, indent=2, default=str))
    (args.output / "git_commit.txt").write_text(_git_commit() + "\n")
    (args.output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    if args.split == "test":
        (args.output / "LOCKED.txt").write_text(
            "This directory is the frozen test-set evaluation output.\n"
            f"git_commit: {_git_commit()}\n"
            f"config_hash: {_config_hash(cfg)}\n"
            f"timestamp_utc: {out['metadata']['run_started_at']}\n"
            "Spec § 6.3: do not regenerate without writing the rationale into the report.\n"
        )

    if not args.no_tables:
        generate_T1(results, args.output / "tables" / "T1_main_results.tex")
        generate_T2(results, args.output / "tables" / "T2_per_class_epet.tex")
    if not args.no_figures:
        generate_F3(results, args.output / "figures" / "F3_tangential_dominance.pdf")
        generate_F4(results, args.output / "figures" / "F4_per_class_epe.pdf")

    print()
    print("=" * 80)
    print(f"  Baseline results on '{args.split}' ({len(eval_ds)} frames)")
    print("=" * 80)
    hdr = f"{'baseline':<32s} {'EPE_all':>8s} {'EPE_dyn':>8s} {'EPE_bg':>8s} {'EPE_t':>8s} {'EPE_r':>8s} {'ang_deg':>8s} {'|Δ|v||':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for name, res in results.items():
        m = res["overall"]
        print(
            f"{name:<32s} "
            f"{m['epe_all']:>8.3f} "
            f"{m['epe_dyn']:>8.3f} "
            f"{m['epe_bg']:>8.3f} "
            f"{m['epe_t']:>8.3f} "
            f"{m['epe_r']:>8.3f} "
            f"{(m['ang_dyn_deg'] if m['ang_dyn_deg']==m['ang_dyn_deg'] else float('nan')):>8.2f} "
            f"{m['mag_err_dyn']:>8.3f}"
        )
    print()
    print(f"[eval] wrote {args.output / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
