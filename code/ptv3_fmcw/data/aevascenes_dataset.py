"""AevaScenes dataset loader producing Pointcept-compatible records.

Spec §3.5 record schema:

    {
      "coord":     (N, 3) float32,
      "feat":      (N, 5) float32,    # [x, y, z, intensity, v_radial]
      "grid_size": 0.1,
      "offset":    (B,) long,         # populated by the collator
      "gt_vxy":    (N, 2) float32,
      "gt_mask":   (N,)   bool,
      "sem_class": (N,)   long,       # eval-only
    }

Spec §3.3: v1 uses `front_wide_lidar` only.
Spec §3.6: voxel size 0.1 m (Pointcept outdoor default).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ptv3_fmcw.data.gt_velocity import Box, per_point_gt_velocity


DEFAULT_LIDAR = "front_wide_lidar"
DEFAULT_GRID_SIZE = 0.1


def _vec3(d: dict) -> np.ndarray:
    return np.array([d["x"], d["y"], d["z"]], dtype=np.float64)


def _quat_wxyz(d: dict) -> np.ndarray:
    """AevaScenes stores rotation as {x, y, z, w}. Return (w, x, y, z)."""
    return np.array([d["w"], d["x"], d["y"], d["z"]], dtype=np.float64)


def parse_box(b: dict) -> Box:
    if b.get("reference_frame", "VEHICLE") != "VEHICLE":
        raise ValueError(
            f"Unexpected box reference_frame={b.get('reference_frame')!r}; "
            "spec assumes VEHICLE"
        )
    return Box(
        center=_vec3(b["pose"]["translation"]),
        dimensions=_vec3(b["dimensions"]),
        rotation=_quat_wxyz(b["pose"]["rotation"]),
        linear_velocity=_vec3(b["linear_velocity"]),
        angular_velocity=_vec3(b["angular_velocity"]),
        track_id=b.get("id", ""),
        cls=b.get("class", ""),
        cls_idx=int(b.get("class_idx", -1)),
    )


@dataclass(frozen=True)
class FrameKey:
    sequence_uuid: str
    frame_idx: int


class AevaScenesFrameDataset:
    """Iterable over (sequence, frame) pairs producing per-point records.

    No torch dependency at construction time so the dataset is usable
    from numpy-only contexts (verification scripts, B0-B3 baselines).
    """

    def __init__(
        self,
        root: str | Path,
        sequence_uuids: Iterable[str],
        lidar: str = DEFAULT_LIDAR,
        grid_size: float = DEFAULT_GRID_SIZE,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        keep_3d_gt: bool = False,
        scene_tag_fn: Callable[[str], str] | None = None,
    ):
        self.root = Path(root)
        self.lidar = lidar
        self.grid_size = grid_size
        self.transform = transform
        self.keep_3d_gt = keep_3d_gt
        self.scene_tag_fn = scene_tag_fn

        self._meta_cache: dict[str, dict] = {}
        self._scene_tag_cache: dict[str, str] = {}
        self._index: list[FrameKey] = []
        for uuid in sequence_uuids:
            seq_path = self.root / uuid / "sequence.json"
            if not seq_path.exists():
                raise FileNotFoundError(seq_path)
            meta = json.loads(seq_path.read_text())
            self._meta_cache[uuid] = meta
            self._scene_tag_cache[uuid] = (
                scene_tag_fn(uuid) if scene_tag_fn is not None else "unknown"
            )
            for f in meta["frames"]:
                self._index.append(FrameKey(uuid, int(f["frame_idx"])))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        key = self._index[idx]
        return self.load(key.sequence_uuid, key.frame_idx)

    def load(self, sequence_uuid: str, frame_idx: int) -> dict[str, Any]:
        meta = self._meta_cache[sequence_uuid]
        frame = meta["frames"][frame_idx]

        pc_path = self.root / sequence_uuid / frame["point_cloud"][self.lidar]["point_cloud_path"]
        npz = np.load(pc_path, allow_pickle=True)

        xyz = npz["xyz"].astype(np.float32)                    # (N, 3)
        v_radial = npz["velocity"].astype(np.float32).reshape(-1)
        intensity = npz["reflectivity"].astype(np.float32).reshape(-1)
        sem_class = npz["semantic_labels_idx"].astype(np.int64).reshape(-1)

        boxes = [parse_box(b) for b in frame.get("boxes", [])]
        gt_v3, gt_mask, _ = per_point_gt_velocity(xyz, boxes)

        feat = np.empty((xyz.shape[0], 5), dtype=np.float32)
        feat[:, :3] = xyz
        feat[:, 3] = intensity
        feat[:, 4] = v_radial

        record: dict[str, Any] = {
            "coord": xyz.copy(),
            "feat": feat,
            "v_radial": v_radial.copy(),     # (N,) signed Doppler, m/s
            "intensity": intensity.copy(),   # (N,) reflectivity proxy
            "grid_size": float(self.grid_size),
            "gt_vxy": gt_v3[:, :2].astype(np.float32),
            "gt_mask": gt_mask,
            "sem_class": sem_class,
            "sequence_uuid": sequence_uuid,
            "frame_idx": int(frame_idx),
            "lidar": self.lidar,
            "scene_tag": self._scene_tag_cache.get(sequence_uuid, "unknown"),
        }
        if self.keep_3d_gt:
            record["gt_v3"] = gt_v3.astype(np.float32)

        if self.transform is not None:
            record = self.transform(record)
        return record

    def list_keys(self) -> list[FrameKey]:
        return list(self._index)

    def metadata(self, sequence_uuid: str) -> dict:
        return self._meta_cache[sequence_uuid]


def list_sequences(root: str | Path) -> list[str]:
    """All sequence UUIDs in the dataset root that have a sequence.json."""
    root = Path(root)
    return sorted(
        p.parent.name for p in root.glob("*/sequence.json") if p.is_file()
    )
