"""Synthetic PF-QA series for the gap-filling post's tests.

The plan's own calendar (four places, 730 days, 24 label dates, the seven windows) on 16 x 16
tiles in 8 x 8 quarters, written as small GeoTIFFs in the dataset's layout. The values are
random, with planted cases: nodata in each layer, same-day codes 0, 900 and -900, every Layer 3
code, a pixel fully actual on no day and one fully actual on every day, days with every pixel
fully actual and with none, label dates with every pixel at 100, and a tie for the example frame.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

ROOT = Path(__file__).resolve().parents[1]
POST = ROOT / "posts" / "2026-11-gap-filling"
SCRIPTS = POST / "scripts"
TILE, QUARTER = 16, 8
GRID = Affine(3.0, 0.0, 5e5, 0.0, -3.0, 3.3e6)
ZONES = {"6466_3380_13": ("48N", 32648), "6730_3430_13": ("50N", 32650),
         "6813_3313_13": ("50N", 32650), "7026_3201_13": ("52N", 32652)}
UNDERSCORE_NAMES = "6813_3313_13"
NEVER_ACTUAL = ("6813_3313_13", (0, 0))
ALWAYS_ACTUAL = ("7026_3201_13", (0, 1))
TIE = (("6466_3380_13", dt.date(2018, 5, 1)), ("6730_3430_13", dt.date(2018, 5, 1)),
       ("6466_3380_13", dt.date(2019, 1, 1)))
ALL_100 = (("6730_3430_13", dt.date(2018, 9, 1)), ("6466_3380_13", dt.date(2019, 3, 1)))


def scripts_on_path():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))


def small_design():
    """The frozen plan's design on 16 x 16 tiles."""
    scripts_on_path()
    import series_plan as SP
    d = SP.design_from(json.loads(SP.PLAN.read_text()))
    q = QUARTER
    return dataclasses.replace(d, tile=TILE, quarter=q,
                               quarters=(("NW", 0, 0), ("NE", 0, q), ("SW", q, 0), ("SE", q, q)))


def make_series(design, seed: int = 0, future: bool = True) -> dict[str, np.ndarray]:
    """{cube: int16 array (days, 3, TILE, TILE)} of layers 1 to 3."""
    rng = np.random.default_rng(seed)
    n, shape = design.n_days, (TILE, TILE)
    labels = {design.index(d) for d in design.label_dates}
    out = {}
    for k, p in enumerate(design.places):
        a = np.zeros((n, 3, TILE, TILE), np.int16)
        for t in range(n):
            u = rng.random()
            actual = np.ones(shape, bool) if u < 0.25 else np.zeros(shape, bool) if u < 0.6 \
                else rng.random(shape) < rng.uniform(0.2, 0.8)
            l1 = np.where(rng.random(shape) < 0.6, 100, rng.integers(2, 100, shape))
            l1 = np.where(actual, 1, l1)
            pick = rng.random(shape)
            l2 = np.select([pick < 0.05, pick < 0.10, pick < 0.15, pick < 0.55, pick < 0.97],
                           [0, 900, -900, -rng.integers(1, 61, shape), rng.integers(1, 61, shape)],
                           np.where(rng.random(shape) < 0.5, -899, 899))
            if not future and t in labels:
                l2 = np.where((l2 >= 1) & (l2 <= 899), -l2, l2)
            l2 = np.where(actual, np.where(rng.random(shape) < 0.95, 0, 900), l2)
            l3 = rng.choice(np.array([-999, 1, 2, 3, 4, 5, 6, 7]), shape)
            l3 = np.where(actual & (rng.random(shape) < 0.9), 1, l3)
            a[t] = np.stack([l1, l2, l3]).astype(np.int16)
        if k == 1:
            for t, layer in ((40, 0), (41, 1), (42, 2), (design.index(dt.date(2018, 7, 1)), 1)):
                a[t, layer, 3, 4] = design.nodata
        out[p.cube] = a
    cube, (r, c) = NEVER_ACTUAL
    out[cube][:, 0, r, c] = 100
    cube, (r, c) = ALWAYS_ACTUAL
    out[cube][:, 0, r, c], out[cube][:, 1, r, c], out[cube][:, 2, r, c] = 1, 0, 1
    for cube, day in TIE:
        t = design.index(day)
        out[cube][t, 0], out[cube][t, 1] = 100, 5 if future else -5
    for cube, day in ALL_100:
        t = design.index(day)
        out[cube][t, 0] = 100
        out[cube][t, 1] = np.where(out[cube][t, 1] == 0, -3, out[cube][t, 1])
    return out


def qa_path(den: Path, cube: str, day: dt.date) -> Path:
    zone = ZONES[cube][0]
    name = day.strftime("%Y_%m_%d" if cube == UNDERSCORE_NAMES else "%Y-%m-%d")
    return den / "planet" / zone / cube / "PF-QA" / f"{name}.tif"


def write_qa(path: Path, layers: np.ndarray, epsg: int, **change) -> None:
    profile = dict(driver="GTiff", width=TILE, height=TILE, count=9, dtype="int16",
                   nodata=-1000, crs=f"EPSG:{epsg}", transform=GRID, compress="lzw")
    profile.update(change)
    path.parent.mkdir(parents=True, exist_ok=True)
    count, dtype = profile["count"], profile["dtype"]
    rows, cols = profile["height"], profile["width"]
    bands = np.zeros((count, rows, cols), dtype)
    bands[:min(3, count)] = layers[:min(3, count), :rows, :cols].astype(dtype)
    with rasterio.open(path, "w", **profile) as f:
        f.write(bands)


def write_dataset(den: Path, design, series: dict) -> None:
    for p in design.places:
        epsg = ZONES[p.cube][1]
        for t, day in enumerate(design.days):
            write_qa(qa_path(den, p.cube, day), series[p.cube][t], epsg)
        for day in design.label_dates:
            sr = qa_path(den, p.cube, day).parent.parent / "PF-SR" / f"{day:%Y-%m-%d}.tif"
            sr.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(sr, "w", driver="GTiff", width=TILE, height=TILE, count=4,
                               dtype="uint16", crs=f"EPSG:{epsg}",
                               transform=GRID) as f:
                f.write(np.full((4, TILE, TILE), 1000, "uint16"))


def write_post1_table(path: Path, design, series: dict) -> None:
    """A table in the form of Post 1's cell_qa_status.parquet, from the synthetic series."""
    import pandas as pd
    rows = []
    for p in design.places:
        for day in design.label_dates:
            l1 = series[p.cube][design.index(day), 0]
            for name, r, c in design.quarters:
                v = l1[r:r + design.quarter, c:c + design.quarter]
                rows.append({"cube": p.cube, "date": pd.Timestamp(day), "quad": name,
                             "fully_observed_cell": bool((v == 1).all()),
                             "fully_synthetic_cell": bool((v == 100).all())})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)


def expected_for(design, series: dict):
    """The run's Expected, with I9's and I10's numbers taken from the synthetic series."""
    scripts_on_path()
    import series_run as R
    frames = {"all_1": 0, "all_100": 0, "other": 0}
    quarters = {"fully_actual": 0, "entirely_filled": 0}
    codes = {"frames": 0, 900: 0, -900: 0}
    for p in design.places:
        for day in design.label_dates:
            l1, l2 = series[p.cube][design.index(day), 0], series[p.cube][design.index(day), 1]
            key = "all_1" if (l1 == 1).all() else "all_100" if (l1 == 100).all() else "other"
            frames[key] += 1
            for _, r, c in design.quarters:
                v = l1[r:r + design.quarter, c:c + design.quarter]
                quarters["fully_actual"] += int((v == 1).all())
                quarters["entirely_filled"] += int((v == 100).all())
            n900, nm900 = int((l2 == 900).sum()), int((l2 == -900).sum())
            codes["frames"] += int(n900 + nm900 > 0)
            codes[900] += n900
            codes[-900] += nm900
    codes["pixels"] = codes[900] + codes[-900]
    return R.Expected(post1_frames=frames, post1_quarters=quarters, same_day_codes=codes)
