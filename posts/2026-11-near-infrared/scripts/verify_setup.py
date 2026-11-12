#!/usr/bin/env python
"""Check the environment, the data layout and the fold geometry, and that a fresh read of the
rasters reproduces the values in artifacts/. Run it before training.

    python posts/2026-11-near-infrared/scripts/verify_setup.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from geoaiwatch import config as C
from geoaiwatch import folds
from geoaiwatch.data import read_sr
from geoaiwatch.labels import label_paths, read_label, remap_to_retained

POST = Path(__file__).resolve().parents[1]
ART = POST / "artifacts"

OK, FAIL = "  ok  ", " FAIL "
status = []


def check(name, fn):
    try:
        detail = fn()
        status.append((True, name, detail or ""))
    except Exception as e:  # noqa: BLE001
        status.append((False, name, f"{type(e).__name__}: {e}"))


def _roots():
    for p in (C.DEN_HOME, C.DATA_ROOT, C.LABEL_ROOT, ART):
        assert p.exists(), f"missing {p}"
    return f"den_home={C.DEN_HOME}  post={POST}"


def _tmpdir():
    if C.TMPDIR is None:
        return "TMPDIR unset — staging to node-local disk unavailable"
    return f"TMPDIR={C.TMPDIR}"


def _gpu():
    import torch
    if not torch.cuda.is_available():
        return "no CUDA visible (expected on a login node)"
    n = torch.cuda.device_count()
    names = {torch.cuda.get_device_name(i) for i in range(n)}
    cap = torch.cuda.get_device_capability(0)
    return f"torch {torch.__version__}  {n}x {', '.join(names)}  sm_{cap[0]}{cap[1]}"


def _frozen():
    f = C.frozen(ART)
    assert f["retained_classes"] == ["agriculture", "forest_veg", "impervious",
                                     "soil", "water"]
    assert len(f["eligible_aois"]) == 4
    assert f["min_test_support_pixels"] == 1024
    assert f["train_recess_px"] == 32
    return f"{len(f['retained_classes'])} classes, {len(f['eligible_aois'])} AOIs"


def _geometry():
    folds.verify()
    n = folds.TRAIN_MASKS["NW"].sum()
    assert n == 691_200, n
    return f"train {n:,} px/image, test {folds.TEST_MASKS['NW'].sum():,}"


def _cubes():
    out = []
    for c in C.frozen(ART)["eligible_cubes"]:
        d = C.cube_dir(c)
        sr = len(list((d / "PF-SR").glob("*.tif")))
        qa = len(list((d / "PF-QA").glob("*.tif")))
        assert sr == 730, f"{c}: {sr} SR files"
        assert qa == 730, f"{c}: {qa} QA files"
        out.append(c)
    return f"{len(out)} cubes, 730 SR + 730 QA each"


def _labels():
    n = sum(1 for p in C.LABEL_ROOT.iterdir() if p.is_dir())
    assert n == 55, f"{n} label dirs"
    for c in C.frozen(ART)["eligible_cubes"]:
        assert len(label_paths(c)) == 24
    return f"{n} training-label dirs, 24 dates per selected cube"


def _artifacts():
    expect = {
        "integrity.parquet": 1320,
        "support_cells.parquet": 5280,
        "frame_qa_summary.parquet": 96,
        "cell_qa_status.parquet": 384,
        "frame_endpoints.parquet": 96,
    }
    for f, n in expect.items():
        p = ART / f
        assert p.exists(), f"missing {f}"
        got = len(pd.read_parquet(p))
        assert got == n, f"{f}: {got} rows, expected {n}"
    return f"{len(expect)} artifacts, row counts match"


def _reproduce_qa():
    """Check that a fresh raster read agrees with cell_qa_status.parquet."""
    cq = pd.read_parquet(ART / "cell_qa_status.parquet")
    cq["date"] = cq["date"].astype(str).str.slice(0, 10)
    checked = 0
    for cube in C.frozen(ART)["eligible_cubes"][:2]:
        d = C.cube_dir(cube)
        for dt in ("2018-03-01", "2019-07-01"):
            with rasterio.open(d / "PF-QA" / f"{dt}.tif") as s:
                l1 = s.read(1)
            assert np.isin(l1, np.arange(1, 101)).all(), f"{cube} {dt}: L1 out of range"
            for q, (r0, c0) in C.QUADS.items():
                v = l1[r0:r0 + C.HALF, c0:c0 + C.HALF]
                row = cq[(cq.cube == cube) & (cq.date == dt) & (cq.quad == q)]
                assert len(row) == 1, f"no artifact row for {cube} {dt} {q}"
                assert bool((v == 1).all()) == bool(row.fully_observed_cell.iloc[0])
                assert bool((v == 100).all()) == bool(row.fully_synthetic_cell.iloc[0])
                checked += 1
    return f"{checked} cells reproduce the committed artifact"


def _band_order():
    """Sanity check, not the basis for the assignment: NIR should dominate."""
    c = C.frozen(ART)["eligible_cubes"][0]
    p = C.cube_dir(c) / "PF-SR" / "2018-07-01.tif"
    x = read_sr(p)                                   # already RGBN
    m = x.mean(axis=(1, 2))
    assert m[3] == m.max(), f"NIR is not the brightest channel: {m.round(4)}"
    return f"RGBN means {m.round(4).tolist()}"


def _label_decode():
    c = C.frozen(ART)["eligible_cubes"][0]
    p = label_paths(c)["2018-03-01"]
    y = read_label(p)
    u = np.unique(y)
    assert set(u) <= set(range(7)) | {C.IGNORE_INDEX}, u
    return f"classes present: {sorted(int(v) for v in u if v != C.IGNORE_INDEX)}"


def _support():
    fz = C.frozen(ART)
    classes, floor = fz["retained_classes"], fz["min_test_support_pixels"]
    counts = []
    for cube in fz["eligible_cubes"]:
        for date, p in label_paths(cube).items():
            y = remap_to_retained(read_label(p), classes)
            for m in folds.TEST_MASKS.values():
                counts.append(np.bincount(y[m & (y != C.IGNORE_INDEX)], minlength=len(classes)))
    counts = np.array(counts)
    assert counts.shape == (384, len(classes)), f"{counts.shape[0]} test cells, expected 384"
    low = int((counts < floor).sum())
    assert low == 0, f"{low} cell x class counts below {floor} pixels"
    return f"all {counts.size} cell x class counts >= {floor} px (smallest {counts.min():,})"


for name, fn in [
    ("roots exist", _roots),
    ("node-local scratch", _tmpdir),
    ("gpu / torch", _gpu),
    ("frozen design", _frozen),
    ("fold geometry", _geometry),
    ("cubes complete", _cubes),
    ("labels complete", _labels),
    ("artifacts present", _artifacts),
    ("QA reproduces artifact", _reproduce_qa),
    ("band order sanity", _band_order),
    ("label decode", _label_decode),
    ("test-cell support", _support),
]:
    check(name, fn)

print()
for ok, name, detail in status:
    print(f"[{OK if ok else FAIL}] {name:26s} {detail}")
print()

n_fail = sum(1 for ok, _, _ in status if not ok)
if n_fail:
    print(f"{n_fail} check(s) failed — resolve before writing training code.")
    sys.exit(1)
print("setup verified.")
