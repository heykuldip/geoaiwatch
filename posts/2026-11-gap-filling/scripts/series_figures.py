#!/usr/bin/env python
"""Draw the post's four figures from the run's committed tables.

    python posts/2026-11-gap-filling/scripts/series_figures.py

Reads runs/ and, for figure 3 alone, the example frame's PF-QA and PF-SR files under DEN_HOME
(geoaiwatch.config); computes no estimand. Each number a figure shows is read from one table and
checked against another, and the numbers the figures print against the values in CAPTIONS; a
failed check stops it with nothing written. Writes the four figures and figures.json
(the commits, the SHA-256 of every file read and written, the software) into figures/. It runs
from a clean git checkout in which runs/ is committed; --without-git, for a copy without git
such as a release archive, records the code files' SHA-256 instead. --replace redraws figures
already in figures/. Layer 2's sources are called earlier, same day or later, and a share prints
as 100% or 0% only when it is exact.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import series_plan as SP
import series_qa as Q
import series_run as R
from matplotlib.colors import Normalize, to_rgba
from matplotlib.patches import Patch
from rasterio.io import MemoryFile

from geoaiwatch import config

POST = SP.POST
REPO = R.REPO
RUNS = R.RUNS
FIGURES = POST / "figures"
FIGS = ("01_daily_strips.png", "02_label_dates.png", "03_example_frame.png", "04_windows.png")
RECORD = "figures.json"
OUTPUTS = FIGS + (RECORD,)
TABLES = ("qa_daily.csv", "qa_monthly.csv", "label_frames.csv", "label_codes.csv",
          "label_summary.csv", "windows.csv", "windows_summary.csv", "window_days.csv")
READS = TABLES + ("inputs.sha256", "audit.json")
CODE_FILES = R.CODE_FILES + (Path(__file__),)

STYLE = {"figure.dpi": 140, "savefig.dpi": 200, "savefig.bbox": "tight",      # Post 1's
         "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "regular"}
ATTRIB = ("Planet Fusion © 2021 Planet Labs Inc. / DynamicEarthNet "
          "(Toker et al. 2022) · CC BY-SA 4.0 · modified")

# Layer 2's sources, the same colors in figures 2 and 3; the plan's past and future are
# called earlier and later in the figures
FULLY_ACTUAL = "#9E9E9E"
SAME_DAY = {"same_day_0": "#009E73", "same_day_900": "#56B4E9", "same_day_minus900": "#0072B2"}
SAME_DAY_WORDS = {"same_day_0": "same day, code 0", "same_day_900": "same day, Sentinel-2 (900)",
                  "same_day_minus900": "same day, Landsat 8 (−900)"}
OFFSETS = "PuOr_r"                                   # earlier purple, later orange
NODATA = "#FFFFFF"
FACING = {"rotation": 270, "va": "bottom"}          # a colorbar's label reads facing the plot


@dataclass(frozen=True)
class Captions:
    """The values the figures' text prints; the render checks each against the tables.
    shares: each place's two-year share of pixel-days fully actual, as printed.
    nodata: nodata pixels at the label dates.
    median_later: the frames' median share of pixels with a later Layer 2 source.
    example: the example frame, the frames tied at its share, the range of its later offsets
             in days, and whether every pixel is entirely filled.
    common: the label dates on which every window is complete, the first, and cells each.
    """
    shares: dict
    nodata: int
    median_later: str
    example: dict
    common: dict


CAPTIONS = Captions(
    shares={"6466_3380_13": "13%", "6730_3430_13": "21%", "6813_3313_13": "25%",
            "7026_3201_13": "36%"},
    nodata=0,
    median_later="10%",
    example={"cube": "6813_3313_13", "date": "2018-01-01", "tied": 13, "offsets": [8, 43],
             "entirely_filled": True},
    common={"dates": 21, "first": "2018-04-01", "cells": 84})


def fail(figure: str, problem) -> None:
    raise Q.CheckFailed(figure, problem)


def pct(num: int, den: int) -> str:
    """A share as the post prints it: whole percentages, one decimal below 1%, and 100% or 0%
    only when exact, otherwise the decimals the remainder needs. Halves round up."""
    if den <= 0 or not 0 <= num <= den:
        raise ValueError(f"not a share: {num}/{den}")
    if num == 0:
        return "0%"
    if num == den:
        return "100%"
    v = Fraction(100 * num, den)
    d = 0 if v >= 1 else 1
    while True:
        n = int(v * 10 ** d + Fraction(1, 2))
        if 0 < n < 100 * 10 ** d:
            whole, rest = divmod(n, 10 ** d)
            return f"{whole}%" if d == 0 else f"{whole}.{rest:0{d}d}%"
        d += 1


def pct_at_least(num: int, den: int, decimals: int = 1) -> str:
    """A share as a lower bound: rounded down at the given decimals, so the printed value is
    never more than the share."""
    if den <= 0 or not 0 <= num <= den:
        raise ValueError(f"not a share: {num}/{den}")
    whole, rest = divmod(100 * num * 10 ** decimals // den, 10 ** decimals)
    return f"{whole}%" if decimals == 0 else f"{whole}.{rest:0{decimals}d}%"


# inputs

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_runs(runs: Path) -> tuple[dict, dict]:
    """The run's files this script reads, each read once: {name: content}, {name: SHA-256}."""
    out, hashes = {}, {}
    for name in READS:
        p = Path(runs) / name
        if not p.is_file():
            fail("inputs", f"{p} not found")
        raw = p.read_bytes()
        hashes[name] = sha256(raw)
        text = raw.decode("utf-8")
        if name.endswith(".csv"):
            out[name] = list(csv.DictReader(text.splitlines()))
        elif name == "audit.json":
            out[name] = json.loads(text)
        else:
            lines = [x.partition("  ") for x in text.splitlines() if x]
            out[name] = {path: h for h, _, path in lines}
    return out, hashes


def committed_runs(runs: Path, repo: Path = REPO) -> str:
    """Every file read from runs/ is tracked; returns the commit that last changed runs/."""
    paths = [str(Path(runs) / n) for n in READS]
    r = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", "--", *paths],
                       capture_output=True, text=True)
    if r.returncode != 0:
        fail("commit", f"runs/ isn't committed: {r.stderr.strip()[:300]}")
    log = subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%H", "--", str(runs)],
                         capture_output=True, text=True, check=True)
    return log.stdout.strip()


def code_files() -> dict:
    out = {}
    for path in CODE_FILES:
        p = Path(path).resolve()
        name = p.relative_to(REPO.resolve()).as_posix() if p.is_relative_to(REPO.resolve()) \
            else p.name
        out[name] = SP.sha256(p)
    return out


# what each figure shows, checked

def daily_strips(design, t: dict, captions: Captions) -> dict:
    """Figure 1: each place's daily share of the tile's pixels fully actual (qa_daily.csv); each
    place's two-year total checked against qa_monthly.csv, its printed share against CAPTIONS."""
    tile = {}
    for r in t["qa_daily.csv"]:
        k = (r["cube"], r["date"])
        fa, px = tile.get(k, (0, 0))
        tile[k] = (fa + int(r["n_fully_actual"]), px + int(r["n_pixels"]))
    days = [d.isoformat() for d in design.days]
    if set(tile) != {(p.cube, d) for p in design.places for d in days}:
        fail("figure 1", "qa_daily.csv doesn't hold each place's days exactly")
    period = f"{design.days[0]:%Y-%m}..{design.days[-1]:%Y-%m}"
    two = {r["cube"]: r for r in t["qa_monthly.csv"] if r["period"] == period}
    out = {}
    for p in design.places:
        fa = np.array([tile[(p.cube, d)][0] for d in days], dtype=np.int64)
        px = np.array([tile[(p.cube, d)][1] for d in days], dtype=np.int64)
        if (px != design.tile ** 2).any():
            fail("figure 1", f"{p.name}: a day without {design.tile ** 2:,} pixels")
        row = two.get(p.cube)
        if row is None or (int(row["n_fully_actual"]), int(row["n_pixel_days"])) != \
                (int(fa.sum()), int(px.sum())):
            fail("figure 1", f"{p.name}: the days don't add up to qa_monthly.csv's {period}")
        share = pct(int(fa.sum()), int(px.sum()))
        if share != captions.shares.get(p.cube):
            fail("figure 1", f"{p.name}: {share} on the figure, "
                             f"{captions.shares.get(p.cube)} in the captions")
        out[p.cube] = {"daily": fa / px, "share": share}
    return out


def label_bars(design, t: dict, captions: Captions) -> dict:
    """Figure 2: each label-date frame as fully actual and filled by Layer 2 source, and nodata;
    each frame's parts add up to its pixels, its sources to its filled pixels, and its nodata
    to that of its quarters in qa_daily.csv."""
    rows = {(r["cube"], r["date"]): r for r in t["label_frames.csv"]}
    want = {(p.cube, d.isoformat()) for p in design.places for d in design.label_dates}
    if set(rows) != want:
        fail("figure 2", "label_frames.csv doesn't hold the label-date frames exactly")
    daily = {}
    for r in t["qa_daily.csv"]:
        k = (r["cube"], r["date"])
        if k in want:
            daily[k] = daily.get(k, 0) + int(r["n_nodata"])
    segments, nodata = {}, 0
    for p in design.places:
        parts = []
        for d in design.label_dates:
            r = {k: int(v) for k, v in rows[(p.cube, d.isoformat())].items() if k.startswith("n_")}
            same = r["n_l2_same_day_0"] + r["n_l2_same_day_900"] + r["n_l2_same_day_minus900"]
            x = (r["n_fully_actual"], r["n_l2_past"], same, r["n_l2_future"], r["n_nodata"])
            if sum(x) != r["n_pixels"] or r["n_pixels"] != design.tile ** 2 or \
                    r["n_l2_past"] + same + r["n_l2_future"] != r["n_filled"]:
                fail("figure 2", f"{p.name} {d}: the parts don't add up")
            if r["n_nodata"] != daily.get((p.cube, d.isoformat())):
                fail("figure 2", f"{p.name} {d}: nodata differs from qa_daily.csv's quarters")
            nodata += r["n_nodata"]
            parts.append(x)
        segments[p.cube] = np.array(parts, dtype=np.int64)
    if nodata != captions.nodata:
        fail("figure 2", f"{nodata:,} nodata pixels, {captions.nodata:,} in the captions")
    return {"segments": segments, "nodata": nodata}


def frame_share(r: dict) -> Fraction:
    return Fraction(int(r["n_l2_future"]), int(r["n_pixels"]))


def ties_from_codes(design, codes: list) -> list:
    """The frames at the largest share of pixels with a later Layer 2 source, counted from
    label_codes.csv's filled pixels by code, in the order the rule's tie-break takes them."""
    l2 = np.array([int(r["l2"]) for r in codes], dtype=np.int64)
    later = Q.sources(l2, np.ones(l2.shape, dtype=bool))["future"]
    n = {}
    for r, is_later in zip(codes, later.tolist()):
        if is_later:
            n[(r["cube"], r["date"])] = n.get((r["cube"], r["date"]), 0) + int(r["n_filled"])
    share = {(p.cube, d.isoformat()): Fraction(n.get((p.cube, d.isoformat()), 0),
                                               design.tile ** 2)
             for p in design.places for d in design.label_dates}
    top = max(share.values())
    return sorted((d, R.cube_number(c), c) for (c, d), s in share.items() if s == top)


def example_frame(design, t: dict, den_root: Path, captions: Captions) -> dict:
    """Figure 3: the frame the plan's rule picks, read from the dataset and checked against
    inputs.sha256, label_frames.csv and label_codes.csv; its disclosure against CAPTIONS."""
    s, frames = t["label_summary.csv"][0], t["label_frames.csv"]
    cube, date = s["example_cube"], s["example_date"]
    if (cube, date) != (captions.example["cube"], captions.example["date"]):
        fail("figure 3", f"label_summary.csv names {cube} {date}; the captions name "
                         f"{captions.example['cube']} {captions.example['date']}")
    top = max(frame_share(r) for r in frames)
    tied = sorted((r["date"], R.cube_number(r["cube"]), r["cube"]) for r in frames
                  if frame_share(r) == top)
    if (tied[0][2], tied[0][0]) != (cube, date):
        fail("figure 3", f"the rule picks {tied[0][2]} {tied[0][0]}, not {cube} {date}")
    if ties_from_codes(design, t["label_codes.csv"]) != tied:
        fail("figure 3", "the frames at the largest share differ between label_frames.csv and "
                         "label_codes.csv")
    median = R.nth([frame_share(r) for r in frames], 5)
    if f"{float(median):.6f}" != s["median_share_future_of_pixels"]:
        fail("figure 3", "the frames' median differs from label_summary.csv's")
    printed = {"tied": len(tied), "median": pct(median.numerator, median.denominator)}
    if printed["tied"] != captions.example["tied"]:
        fail("figure 3", f"{printed['tied']} frames tied, {captions.example['tied']} in "
                         "the captions")
    if printed["median"] != captions.median_later:
        fail("figure 3", f"median {printed['median']}, {captions.median_later} in "
                         "the captions")

    day, planet, hashes = dt.date.fromisoformat(date), Path(den_root) / "planet", t["inputs.sha256"]
    try:
        found = Q.files_by_day(config.cube_dir(cube, planet) / "PF-QA")[0].get(day, [])
    except FileNotFoundError as e:
        fail("figure 3", str(e))
    if len(found) != 1:
        fail("figure 3", f"{cube} {date}: {len(found)} PF-QA files")
    qa_rel = R.relative(found[0], den_root)
    digest, l1, l2, l3 = Q.read_day(found[0], design, None)
    if hashes.get(qa_rel) != digest:
        fail("figure 3", f"{qa_rel}: SHA-256 {digest}, not as in inputs.sha256")
    sr = Q.find_pf_sr(planet, cube, day)
    sr_rel, raw = R.relative(sr, den_root), sr.read_bytes()
    seen = t["audit.json"]["example_frame"]
    if not (hashes.get(sr_rel) == sha256(raw) == seen["pf_sr_check"]["sha256"]
            and seen["pf_sr"] == sr_rel):
        fail("figure 3", f"{sr_rel}: not the PF-SR file the run hashed")
    with MemoryFile(raw) as m, m.open() as f:
        rgb = f.read([3, 2, 1])                          # red, green, blue (disk order B, G, R, N)

    c = Q.classify(l1, l2, l3, design.nodata)
    src = Q.sources(l2, c.filled)
    mine = {"n_pixels": l2.size, "n_nodata": c.nodata.sum(), "n_fully_actual": c.fully_actual.sum(),
            "n_filled": c.filled.sum(), "n_entirely_filled": c.entirely_filled.sum(),
            "n_blended": c.blended.sum(), **{f"n_l2_{k}": v.sum() for k, v in src.items()}}
    row = next(r for r in frames if (r["cube"], r["date"]) == (cube, date))
    if {k: int(v) for k, v in mine.items()} != {k: int(row[k]) for k in mine}:
        fail("figure 3", "the frame's pixels differ from label_frames.csv")
    codes, n = np.unique(l2[c.filled], return_counts=True)
    table = {}
    for r in t["label_codes.csv"]:
        if (r["cube"], r["date"]) == (cube, date):
            table[int(r["l2"])] = table.get(int(r["l2"]), 0) + int(r["n_filled"])
    if dict(zip(codes.tolist(), n.tolist())) != table:
        fail("figure 3", "the frame's Layer 2 codes differ from label_codes.csv")
    later = l2[src["future"]]
    printed |= {"offsets": [int(later.min()), int(later.max())] if later.size else None,
                "entirely_filled": bool(c.entirely_filled.all()),
                "share": pct(int(src["future"].sum()), l2.size),
                "first_day": day == design.days[0]}
    for k in ("offsets", "entirely_filled"):
        if printed[k] != captions.example[k]:
            fail("figure 3", f"{k}: {printed[k]}, {captions.example[k]} in the captions")
    return {"cube": cube, "date": date, "rgb": rgb, "l2": l2, "classes": c, "sources": src,
            "printed": printed, "n_frames": len(frames),
            "dataset": {qa_rel: digest, sr_rel: sha256(raw)}}


def window_cells(design, t: dict, captions: Captions) -> dict:
    """Figure 4: each complete cell's covered share, on the label dates where every window is
    complete; each window's cells checked against windows_summary.csv's common rows, and each
    cell's uncovered pixels against its pixels with no fully-actual day in window_days.csv, so
    a quarter counts under its window only when both tables agree that every pixel is covered."""
    rows = [r for r in t["windows.csv"] if r["complete"] == "true"]
    dates = {w.name: {r["date"] for r in rows if r["window"] == w.name} for w in design.windows}
    common = sorted(set.intersection(*dates.values()))
    if (len(common), common[0] if common else None) != (captions.common["dates"],
                                                         captions.common["first"]):
        fail("figure 4", f"{len(common)} common label dates from "
                         f"{common[0] if common else None}; the captions say "
                         f"{captions.common['dates']} from {captions.common['first']}")
    total, zero = {}, {}
    for r in t["window_days.csv"]:
        k = (r["cube"], r["date"], r["quarter"], r["window"])
        total[k] = total.get(k, 0) + int(r["n_pixels"])
        if int(r["fully_actual_days"]) == 0:
            zero[k] = zero.get(k, 0) + int(r["n_pixels"])
    by_days = {w.name: {k[1] for k in total if k[3] == w.name} for w in design.windows}
    if sorted(set.intersection(*by_days.values())) != common:
        fail("figure 4", "the common label dates differ between windows.csv and window_days.csv")
    summary = {(r["cube"], r["window"]): r for r in t["windows_summary.csv"]
               if r["dates"] == "common"}
    cells = {}
    for p in design.places:
        for w in design.windows:
            here = sorted((r for r in rows if r["cube"] == p.cube and r["window"] == w.name
                           and r["date"] in common), key=lambda r: (r["date"], r["quarter"]))
            cov = np.array([int(r["n_covered"]) for r in here], dtype=np.int64)
            px = np.array([int(r["n_pixels"]) for r in here], dtype=np.int64)
            sm = summary.get((p.cube, w.name))
            if len(here) != captions.common["cells"] or sm is None or \
                    (int(sm["n_covered"]), int(sm["n_pixels"]), int(sm["n_cells"]),
                     int(sm["n_label_dates"])) != (int(cov.sum()), int(px.sum()), len(here),
                                                   len(common)):
                fail("figure 4", f"{p.name} {w.name}: the cells differ from "
                                 f"windows_summary.csv's common row")
            keys = [(r["cube"], r["date"], r["quarter"], r["window"]) for r in here]
            if [total.get(k) for k in keys] != px.tolist() or \
                    [zero.get(k, 0) for k in keys] != (px - cov).tolist():
                fail("figure 4", f"{p.name} {w.name}: the cells' pixels with no fully-actual "
                                 f"day differ from window_days.csv's")
            cells[(p.cube, w.name)] = {"shares": cov / px, "pooled": cov.sum() / px.sum(),
                                       "full": int((cov == px).sum()), "n": len(here)}
    return {"common": common, "cells": cells}


# drawing

def footer(fig, text: str, room: float = 0.10) -> None:
    fig.text(0.5, 0.01, text, ha="center", va="bottom", fontsize=6.5, color="#555555", wrap=True)
    fig.get_layout_engine().set(rect=(0, room, 1, 1 - room))


TICKS = [0, 0.25, 0.5, 0.75, 1]
TICK_WORDS = [pct(k, 4) for k in range(5)]              # 0% to 100%, each exact


def percent_axis(ax) -> None:
    ax.set_yticks(TICKS, TICK_WORDS)


def true_color(rgb: np.ndarray) -> np.ndarray:
    """Post 1's display: each band stretched between its 1st and 99th percentiles, gamma 0.75."""
    a = rgb.astype(np.float64) / config.SR_SCALE
    out = np.empty_like(a)
    for i in range(3):
        lo, hi = np.percentile(a[i], [1, 99])
        out[i] = np.clip((a[i] - lo) / max(hi - lo, 1e-9), 0, 1)
    return (out ** 0.75).transpose(1, 2, 0)


def layer2_map(l2: np.ndarray, c, src: dict) -> tuple[np.ndarray, Normalize]:
    """RGBA: fully actual one gray; same-day codes their own colors; earlier and later offsets
    on a diverging scale symmetric about zero to the frame's largest offset; nodata white."""
    rgba = np.empty(l2.shape + (4,))
    rgba[:] = to_rgba(NODATA)
    rgba[c.fully_actual] = to_rgba(FULLY_ACTUAL)
    for k, color in SAME_DAY.items():
        rgba[src[k]] = to_rgba(color)
    off = src["past"] | src["future"]
    m = int(np.abs(l2[off]).max()) if off.any() else 1
    norm = Normalize(-m, m)
    rgba[off] = mpl.colormaps[OFFSETS](norm(l2[off].astype(np.float64)))
    return rgba, norm


def month_ticks(days: list) -> tuple[list, list]:
    marks = [(i, d) for i, d in enumerate(days) if d.day == 1 and d.month in (1, 7)]
    return [i for i, _ in marks], [f"{d:%Y-%m}" for _, d in marks]


def draw_strips(design, strips: dict, path: Path) -> None:
    n = design.n_days
    fig, axes = plt.subplots(len(design.places), 1, figsize=(9, 1.0 * len(design.places) + 1.6),
                             sharex=True, layout="constrained")
    marks = [design.index(d) for d in design.label_dates]
    for ax, p in zip(axes, design.places):
        im = ax.imshow(strips[p.cube]["daily"][None, :], aspect="auto", cmap="cividis", vmin=0,
                       vmax=1, interpolation="none", extent=(-0.5, n - 0.5, 0, 1))
        ax.set_xticks(marks, minor=True)
        ax.tick_params(axis="x", which="minor", top=True, bottom=False, length=4, width=0.8,
                       color="#333333")
        ax.set_yticks([])
        ax.set_title(f"{p.name} · {strips[p.cube]['share']} of pixel-days fully actual",
                     loc="left", fontsize=9, pad=9)
    ax.set_xticks(*month_ticks(design.days))
    cb = fig.colorbar(im, ax=list(axes), shrink=0.9, pad=0.01)
    cb.set_label("share of the image's pixels fully actual", **FACING)
    cb.set_ticks(TICKS, labels=TICK_WORDS)
    footer(fig, f"Each column is one day, {design.days[0]} to {design.days[-1]}; the ticks above "
                f"each strip mark the {len(design.label_dates)} map dates. An image is "
                f"{design.tile:,} × {design.tile:,} pixels at 3 m.\n{ATTRIB}", room=0.12)
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)


def draw_bars(design, bars: dict, path: Path) -> None:
    colors = [FULLY_ACTUAL, mpl.colormaps[OFFSETS](0.15), SAME_DAY["same_day_0"],
              mpl.colormaps[OFFSETS](0.85), NODATA]
    words = ["fully actual", "filled: Layer 2 source earlier", "filled: same day",
             "filled: later", "nodata"]
    parts = 5 if bars["nodata"] else 4
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharey=True, layout="constrained")
    x = np.arange(len(design.label_dates))
    ticks = [(i, d) for i, d in enumerate(design.label_dates) if d.month in (1, 7)]
    for ax, p in zip(axes.flat, design.places):
        seg = bars["segments"][p.cube] / design.tile ** 2
        bottom = np.zeros(len(x))
        for k in range(parts):
            ax.bar(x, seg[:, k], bottom=bottom, width=0.85, color=colors[k],
                   edgecolor="#777777" if k == 4 else "none", linewidth=0.4)
            bottom += seg[:, k]
        ax.set_title(p.name, loc="left")
        ax.set_xticks([i for i, _ in ticks], [f"{d:%Y-%m}" for _, d in ticks])
        ax.set_xlim(-0.6, len(x) - 0.4)
        ax.set_ylim(0, 1)
        percent_axis(ax)
    fig.legend(handles=[Patch(facecolor=colors[k], edgecolor="#777777" if k == 4 else "none",
                              label=words[k]) for k in range(parts)],
               loc="lower center", ncol=parts, frameon=False, bbox_to_anchor=(0.5, 0.075))
    none = nodata_words(bars["nodata"])
    none = f" {none}" if none else ""
    n = len(design.places) * len(design.label_dates)
    footer(fig, f"The {n} images on map dates, {len(design.label_dates)} a place, each "
                f"{design.tile:,} × {design.tile:,} pixels. For a filled pixel, Layer 2 records "
                f"one source: earlier, the same day or later.{none}\n{ATTRIB}", room=0.15)
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)


def nodata_words(k: int) -> str:
    """Figure 2's sentence on the pixels that carry the nodata code in the quality layers;
    none when no pixel does."""
    if k == 0:
        return ""
    who = "No pixel carries" if k == 0 else "1 pixel carries" if k == 1 else f"{k:,} pixels carry"
    return f"{who} the nodata code in quality layers 1–3."


def frame_caption(design, place: str, f: dict) -> str:
    p = f["printed"]
    lo_hi = p["offsets"]
    filled = "Every pixel is entirely gap-filled, and" if p["entirely_filled"] else "For"
    later = f"{p['share']} of its pixels" if p["share"] != "100%" else "every pixel"
    span = "" if lo_hi is None else (f", {lo_hi[0]} days later" if lo_hi[0] == lo_hi[1]
                                     else f", {lo_hi[0]} to {lo_hi[1]} days later")
    lead = (f"{filled} Layer 2 records a later source for {later}{span}." if p["entirely_filled"]
            else f"{filled} {later}, Layer 2 records a later source{span}.")
    if p["tied"] > 1:
        tie = (f" {p['tied']} of the {f['n_frames']} images are at {p['share']}; the rule's "
               f"tie-break, the earliest map date, picked this one"
               + (", the first day of the series." if p["first_day"] else "."))
    else:
        tie = ""
    return "\n".join([
        f"{place} · {f['date']} · the whole image, {design.tile:,} × {design.tile:,} pixels "
        "at 3 m.",
        lead,
        "Chosen by a rule fixed before the run: the image with the largest share of its pixels "
        "filled and with a later Layer 2 source.",
        f"{tie.strip()} Across the {f['n_frames']} images, the median share is {p['median']}."
        .strip(),
        ATTRIB])


def frame_handles(c, src: dict) -> list:
    """Figure 3's legend: the gray and same-day colors the image holds, and no others."""
    handles = [Patch(facecolor=FULLY_ACTUAL, label="fully actual")] if c.fully_actual.any() else []
    return handles + [Patch(facecolor=SAME_DAY[k], label=SAME_DAY_WORDS[k])
                      for k in SAME_DAY if src[k].any()]


def draw_frame(design, f: dict, place: str, path: Path) -> None:
    rgba, norm = layer2_map(f["l2"], f["classes"], f["sources"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 6.0), layout="constrained")
    ax[0].imshow(true_color(f["rgb"]), interpolation="none")
    ax[0].set_title("True color" + (", every pixel gap-filled"
                                    if f["printed"]["entirely_filled"] else ""))
    ax[1].imshow(rgba, interpolation="none")
    ax[1].set_title("Layer 2's recorded source")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=OFFSETS), ax=ax[1], shrink=0.8,
                      pad=0.02)
    cb.set_label("days earlier (−) or later (+)", **FACING)
    handles = frame_handles(f["classes"], f["sources"])
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
                   bbox_to_anchor=(0.5, 0.115))
    footer(fig, frame_caption(design, place, f), room=0.19 if handles else 0.15)
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)


def draw_windows(design, w: dict, path: Path) -> None:
    order = [x for x in design.windows if x.name.startswith("nested")] + \
        [x for x in design.windows if x.name.startswith("block")]
    nested = sum(x.name.startswith("nested") for x in order)
    pos = [i + (0.8 if i >= nested else 0) for i in range(len(order))]
    names = [f"{x.days} d" if x.name.startswith("nested") else f"{x.last}–{x.first}"
             for x in order]
    rng = np.random.default_rng(0)                          # the dots' spread, the same each time
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharey=True, layout="constrained")
    for ax, p in zip(axes.flat, design.places):
        labels = []
        for x0, win, name in zip(pos, order, names):
            cell = w["cells"][(p.cube, win.name)]
            ax.scatter(x0 + rng.uniform(-0.28, 0.28, cell["n"]), cell["shares"], s=7,
                       color="#4D4D4D", alpha=0.55, linewidths=0)
            ax.hlines(cell["pooled"], x0 - 0.36, x0 + 0.36, color="#D55E00", linewidth=1.8)
            labels.append(f"{name}\n{cell['full']}/{cell['n']}")
        ax.set_xticks(pos, labels, fontsize=7.5)
        ax.set_title(p.name, loc="left")
        ax.set_ylim(-0.03, 1.03)
        percent_axis(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("pixels with a fully-actual day")
    n = w["cells"][(design.places[0].cube, order[0].name)]["n"]
    footer(fig, "\n".join([
        f"Windows ending the day before the map date, {order[0].days} to "
        f"{order[nested - 1].days} days long; then the three 30-day blocks before it, in days "
        f"before the map date.",
        f"Each dot is a quarter of an image on one of the {len(w['common'])} map dates from "
        f"{w['common'][0]}, on which every window is complete; the orange line pools the pixels "
        f"of all {n}.",
        "Under each window: the quarters in which every pixel has a fully-actual day, not "
        "necessarily the same one.",
        ATTRIB]), room=0.14)
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)


# the render

def render(design, runs: Path, den_root: Path, out: Path, captions: Captions = CAPTIONS,
           git=None, replace: bool = False, log=print) -> dict:
    """The four figures and figures.json; raises CheckFailed with nothing written."""
    out = Path(out)
    existing = [n for n in OUTPUTS if (out / n).exists()]
    if existing and not replace:
        fail("outputs", f"{out} already holds {existing}; --replace redraws them")
    if git:
        state = git()
        R.check_commit(state, "before the figures")
        runs_commit = committed_runs(runs)
    else:
        state, runs_commit = {"commit": None, "clean": None, "git": "not used"}, None
    code = code_files()
    t, hashes = read_runs(runs)
    strips = daily_strips(design, t, captions)
    bars = label_bars(design, t, captions)
    frame = example_frame(design, t, den_root, captions)
    cells = window_cells(design, t, captions)
    place = next(p.name for p in design.places if p.cube == frame["cube"])
    tmp = out.parent / f".figures-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        with mpl.rc_context(STYLE):
            draw_strips(design, strips, tmp / FIGS[0])
            draw_bars(design, bars, tmp / FIGS[1])
            draw_frame(design, frame, place, tmp / FIGS[2])
            draw_windows(design, cells, tmp / FIGS[3])
        if git and git() != state:
            fail("commit", "the checkout changed during the render")
        if code_files() != code:
            fail("commit", "the code changed during the render")
        record = {
            "figures": list(FIGS),
            "code": {**state, "files": code},
            "runs": {"commit": runs_commit, "files": hashes},
            "dataset": frame["dataset"],
            "captions": asdict(captions),
            "checks": {**{k: "passed" for k in ("inputs", "figure 1", "figure 2", "figure 3",
                                                "figure 4")},
                       "commit": "passed" if git else "not checked (git not used)"},
            "software": {**R.versions(), "matplotlib": mpl.__version__},
            "outputs": {n: SP.sha256(tmp / n) for n in FIGS},
        }
        (tmp / RECORD).write_text(json.dumps(record, indent=1) + "\n")
        out.mkdir(parents=True, exist_ok=True)
        for n in OUTPUTS:
            os.replace(tmp / n, out / n)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"every check passed; {len(OUTPUTS)} files written to {out.name}/")
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--without-git", action="store_true",
                    help="for a copy without git: record the code files' SHA-256, not a commit")
    ap.add_argument("--replace", action="store_true", help="redraw figures already in figures/")
    a = ap.parse_args(argv)
    try:
        design, _ = SP.load()
    except SP.PlanError as e:
        print(f"plan: {e}")
        return 1
    try:
        render(design, RUNS, config.DEN_HOME, FIGURES,
               git=None if a.without_git else R.git_state, replace=a.replace)
    except Q.CheckFailed as e:
        print(f"stopped: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
