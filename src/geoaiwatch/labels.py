"""Label decoding.

DynamicEarthNet labels are seven-band one-hot rasters valued 0 or 255. A pixel gets a class
only if exactly one band is set, and IGNORE_INDEX otherwise; 242 of the 1,320 training files
have pixels that are not one-hot.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import rasterio

from .config import CLASS_BANDS, EXPECTED_DATES, IGNORE_INDEX, label_dir

DATE_RE = re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})")


def parse_date(p: Path) -> str | None:
    m = DATE_RE.search(p.stem)
    return "-".join(m.groups()) if m else None


def label_paths(cube: str) -> dict[str, Path]:
    """{date: path} for one cube, asserting the exact expected date set."""
    d = label_dir(cube)
    out = {}
    for p in sorted(d.rglob("*.tif")):
        dt = parse_date(p)
        if dt is None:
            raise ValueError(f"unparseable date: {p}")
        if dt in out:
            raise ValueError(f"duplicate date {dt} in {d}")
        out[dt] = p
    missing = sorted(set(EXPECTED_DATES) - set(out))
    extra = sorted(set(out) - set(EXPECTED_DATES))
    if missing or extra:
        raise ValueError(f"{cube}: missing={missing} extra={extra}")
    return out


def read_label(path: Path, validate: bool = True) -> np.ndarray:
    """Seven-band one-hot to an int16 class index (0..6), IGNORE_INDEX elsewhere."""
    with rasterio.open(path) as s:
        a = s.read()

    if validate:
        assert a.shape[0] == 7, f"{path}: {a.shape[0]} bands"
        assert a.dtype == np.uint8, f"{path}: dtype {a.dtype}"
        assert a.shape[1:] == (1024, 1024), f"{path}: shape {a.shape[1:]}"
        assert np.isin(a, (0, 255)).all(), f"{path}: values outside {{0,255}}"

    hot = a == 255
    n = hot.sum(axis=0)
    lab = np.where(n == 1, hot.argmax(axis=0), IGNORE_INDEX)
    return lab.astype(np.int16)


def label_stats(path: Path) -> dict:
    """Per-file integrity record, matching artifacts/integrity.parquet."""
    with rasterio.open(path) as s:
        a = s.read()
    hot = a == 255
    n = hot.sum(axis=0)
    out = {
        "bands": int(a.shape[0]),
        "dtype": str(a.dtype),
        "h": int(a.shape[1]),
        "w": int(a.shape[2]),
        "vals_ok": bool(np.isin(a, (0, 255)).all()),
        "n_zero_hot": int((n == 0).sum()),
        "n_multi_hot": int((n > 1).sum()),
    }
    out.update({CLASS_BANDS[i]: int(hot[i].sum()) for i in range(a.shape[0])})
    return out


def remap_to_retained(lab: np.ndarray, retained: list[str]) -> np.ndarray:
    """Map the seven-class index to the retained classes, renumbered 0..k-1 in order."""
    out = np.full_like(lab, IGNORE_INDEX)
    for new, name in enumerate(retained):
        out[lab == CLASS_BANDS.index(name)] = new
    return out
