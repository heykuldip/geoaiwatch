"""Paths and constants shared by the posts.

DEN_HOME points to the DynamicEarthNet download and defaults to data/DynamicEarthNet in the
repository. Each post keeps its design values in its own artifacts/ folder.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

# roots

_REPO = Path(__file__).resolve().parents[2]
DEN_HOME = Path(os.environ.get("DEN_HOME", _REPO / "data" / "DynamicEarthNet"))

DATA_ROOT = DEN_HOME / "planet"                # zones and labels
LABEL_ROOT = DATA_ROOT / "labels"              # 55 training-label directories

# Node-local disk if the scheduler sets TMPDIR; frames are staged there rather
# than re-read from DATA_ROOT every epoch.
TMPDIR = Path(os.environ["TMPDIR"]) if os.environ.get("TMPDIR") else None

# label schema  (DynamicEarthNet, 7-band one-hot uint8 {0,255})

CLASS_BANDS = [
    "impervious", "agriculture", "forest_veg",
    "wetlands", "soil", "water", "snow_ice",
]
IGNORE_INDEX = 255

EXPECTED_DATES = [f"{y}-{m:02d}-01" for y in (2018, 2019) for m in range(1, 13)]

# PF-SR imagery
# Band order: the GeoTIFF on disk is B, G, R, NIR (Planet's 4-band order, per
# the March 2021 Planet Fusion specification). The DynamicEarthNet loader
# reorders to R, G, B, NIR, and its published normalization statistics match
# the reordered tensor, not the file; they are kept for reference.

SR_SCALE = 10_000
DISK_ORDER = ("B", "G", "R", "N")
MODEL_ORDER = ("R", "G", "B", "N")
BGRN_TO_RGBN = (2, 1, 0, 3)                    # index into the disk array

PF_MEANS_RGBN = (1042.59, 915.62, 671.26, 2605.21)   # corpus-wide, reference only
PF_STDS_RGBN = (957.96, 715.55, 596.94, 1059.90)     # statistics are computed per fold

# PF-QA  (Planet_fusion_specification_March_2021)
# Layer 1  1 = no gap filling, 100 = entirely gap filled (March 2021
#          specification; later versions use 0 for no gap filling).
# Layer 2  valid range -900..900; +900 = same-day Sentinel-2, -900 = same-day
#          Landsat-8, both zero days distant. -999 belongs to Layers 3-5.
# Layer 3  1 clear, 2-6 cloud-related, 7 band misalignment, -999 no PS scene.

QA_SPEC_VERSION = "Planet_fusion_specification_March_2021"
L1_VALID = (1, 100)
L2_VALID = (-900, 900)
L2_SAME_DAY = (900, -900)
L3_MEANING = {
    1: "clear", 2: "bright_clouds", 3: "shadows", 4: "other_cloud",
    5: "adjacent_clouds", 6: "cross_scene", 7: "misalign", -999: "no_scene",
}

# fold geometry  (see a post's artifacts/screen_config.json)

TILE = 1024
HALF = 512
TRAIN_RECESS_PX = 32
QUADS = {"NW": (0, 0), "NE": (0, HALF), "SW": (HALF, 0), "SE": (HALF, HALF)}

# canonical AOI key
# Identifiers appear as both '6466_3380_13_48N' and '7026_3201_13-52N';
# keying on the bare cube id handles both.

_ZONE_RE = re.compile(r"[-_](\d{2}[NS])$")


def strip_zone(s: str) -> str:
    return _ZONE_RE.sub("", s)


def zone_of(s: str) -> str | None:
    m = _ZONE_RE.search(s)
    return m.group(1) if m else None


# frozen design, read from artifacts

def load_screen(art: Path) -> tuple[dict, dict]:
    """Return the screen's config and result from a post's artifacts folder."""
    cfg = json.loads((art / "screen_config.json").read_text())
    res = json.loads((art / "screen_result.json").read_text())
    return cfg, res


def frozen(art: Path):
    cfg, res = load_screen(art)
    return dict(
        retained_classes=res["retained_classes"],
        eligible_aois=res["eligible_aois"],
        eligible_cubes=[strip_zone(a) for a in res["eligible_aois"]],
        min_test_support_pixels=cfg["min_test_support_pixels"],
        train_recess_px=cfg["fold_geometry"]["train_recess_px"],
    )


def cube_dir(cube: str, root: Path = DATA_ROOT) -> Path:
    """Locate a cube directory under the zone tree."""
    hits = [p for p in root.rglob(cube) if p.is_dir()]
    if len(hits) != 1:
        raise FileNotFoundError(f"{cube}: found {len(hits)} matches under {root}")
    return hits[0]


def label_dir(cube: str, root: Path = LABEL_ROOT) -> Path:
    """Label directory for a cube, tolerating both '_48N' and '-52N'."""
    hits = [p for p in root.iterdir() if p.is_dir() and strip_zone(p.name) == cube]
    if len(hits) != 1:
        raise FileNotFoundError(f"{cube}: found {len(hits)} label dirs under {root}")
    return hits[0]
