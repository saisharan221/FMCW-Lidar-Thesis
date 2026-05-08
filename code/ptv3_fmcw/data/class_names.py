"""AevaScenes per-point semantic class index <-> name mapping.

Recovered from the data on disk (30 sequences x 3 frames sampled
2026-05-05). The set is stable across the full dataset; ids 8, 10, 11
are not observed in the AevaScenes v0.2 release.

Dynamic-vs-static partition follows the data-exploration grouping:
roads, vegetation, buildings, etc. are static; vehicles, pedestrians,
cyclists are dynamic. Static classes are expected to carry GT velocity
near zero by construction (the velocity field is ego-compensated).
"""

SEMANTIC_NAMES: dict[int, str] = {
    0: "unknown",
    1: "car",
    2: "bus",
    3: "truck",
    4: "trailer",
    5: "vehicle_on_rails",
    6: "other_vehicle",
    7: "bicycle",
    9: "pedestrian",
    12: "bicyclist",
    13: "traffic_item",
    14: "traffic_sign",
    15: "pole_trunk",
    16: "building",
    17: "other_structure",
    18: "vegetation",
    19: "road",
    20: "lane_boundary",
    21: "road_marking",
    22: "reflective_marker",
    23: "sidewalk",
    24: "other_ground",
}

DYNAMIC_CLASS_IDS: frozenset[int] = frozenset(
    {1, 2, 3, 4, 5, 6, 7, 9, 12}  # vehicles + cyclists + pedestrians
)

# Top-10 classes for per-class reporting (data_exploration §1 ranking).
TOP_CLASSES: tuple[str, ...] = (
    "pole_trunk",
    "car",
    "traffic_sign",
    "traffic_item",
    "pedestrian",
    "unknown",
    "truck",
    "other_structure",
    "bicycle",
    "bus",
)


def name_of(idx: int) -> str:
    return SEMANTIC_NAMES.get(int(idx), f"class_{int(idx)}")


def is_dynamic(idx: int) -> bool:
    return int(idx) in DYNAMIC_CLASS_IDS
