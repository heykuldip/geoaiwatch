#!/usr/bin/env python
"""Run the post's frozen plan: how much of DynamicEarthNet's daily series is real.

    python posts/2026-11-gap-filling/scripts/series_run.py                  # the run
    python posts/2026-11-gap-filling/scripts/series_run.py --check-inputs   # inputs only

DEN_HOME points to the dataset (geoaiwatch.config). The run reads every PF-QA file of the four
places once, asserts the plan's checks I1 to I13 and four more (pf_sr, blocks_sum, gap_days and
tables), and writes runs/ only when every check has passed; a failed check stops it and leaves
runs/ as it was. It runs from a clean git checkout and records its commit; --without-git, for a
copy without git such as a release archive, records the code files' SHA-256 instead.
--check-inputs runs I1 to I3 and records the files' metadata, computing no estimand, and writes
runs/smoke/input_check.json.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
import rasterio
import series_plan as SP
import series_qa as Q

from geoaiwatch import config

POST = SP.POST
REPO = POST.parents[1]
RUNS = POST / "runs"
POST1_CELLS = POST.parent / "2026-11-near-infrared" / "artifacts" / "cell_qa_status.parquet"
CODE_FILES = (SP.POST / "scripts" / "series_plan.py", SP.POST / "scripts" / "series_qa.py",
              SP.POST / "scripts" / "series_run.py", Path(config.__file__))

TABLES = ("qa_daily.csv", "qa_monthly.csv", "gaps.csv", "gap_lengths.csv", "label_frames.csv",
          "label_cells.csv", "label_codes.csv", "label_summary.csv", "windows.csv",
          "window_days.csv", "block_coverage.csv", "windows_summary.csv")
OUTPUTS = TABLES + ("expectations.json", "inputs.sha256", "audit.json")


@dataclass(frozen=True)
class Expected:
    """The numbers the checks compare with, from the plan's text."""
    complete_dates: dict = field(default_factory=lambda: dict(SP.COMPLETE_DATES))
    complete_cells: dict = field(default_factory=lambda: dict(SP.COMPLETE_CELLS))
    short_windows: tuple = SP.SHORT_WINDOWS
    post1_frames: dict = field(default_factory=lambda: dict(SP.POST1_FRAMES))
    post1_quarters: dict = field(default_factory=lambda: dict(SP.POST1_QUARTERS))
    same_day_codes: dict = field(default_factory=lambda: dict(SP.SAME_DAY_CODES))
    rows: dict = field(default_factory=lambda: dict(SP.ROWS))


# checks across the run

def check_windows(design, expected: Expected) -> dict[str, list[dt.date]]:
    """I8: each window's length and end, and the complete windows' numbers."""
    complete = {}
    for w in design.windows:
        dates = []
        for d in design.label_dates:
            first, last = d - dt.timedelta(days=w.first), d - dt.timedelta(days=w.last)
            if (last - first).days + 1 != w.days or last >= d or w.last < 1:
                raise Q.CheckFailed("I8", f"{w.name} before {d}: {first}..{last}")
            if design.first_day <= first and last <= design.last_day:
                dates.append(d)
        complete[w.name] = dates
    cells = len(design.places) * len(design.quarters)
    for w in design.windows:
        kind = "short" if w.name in expected.short_windows else "long"
        n = len(complete[w.name])
        if n != expected.complete_dates[kind] or n * cells != expected.complete_cells[kind]:
            raise Q.CheckFailed("I8", f"{w.name}: {n} complete label dates ({n * cells} cells), "
                                      f"the plan says {expected.complete_dates[kind]} "
                                      f"({expected.complete_cells[kind]})")
    return complete


def day_classes(tile_fully_actual: list[int], design) -> list[str]:
    total = design.tile ** 2
    return ["all" if n == total else "none" if n == 0 else "some" for n in tile_fully_actual]


def check_day_classes(results, design) -> None:
    """I6: all + some + none = the series' days, for every place."""
    for r in results:
        classes = day_classes(r.tile_fully_actual, design)
        n = {k: classes.count(k) for k in ("all", "some", "none")}
        if len(classes) != design.n_days or sum(n.values()) != design.n_days:
            raise Q.CheckFailed("I6", f"{r.cube}: {n} over {len(classes)} days")


def read_post1_cells(path: Path) -> dict:
    """Post 1's quarter flags at the label dates, keyed by (cube, date, quarter)."""
    import pandas as pd
    cols = ["cube", "date", "quad", "fully_observed_cell", "fully_synthetic_cell"]
    df = pd.read_parquet(path, columns=cols)
    keys = list(zip(df["cube"].astype(str).map(config.strip_zone),
                    pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"), df["quad"].astype(str)))
    if len(set(keys)) != len(keys):
        raise Q.CheckFailed("I9", f"{path.name} repeats a quarter")
    flags = zip(df["fully_observed_cell"].astype(bool), df["fully_synthetic_cell"].astype(bool))
    return {k: (bool(a), bool(b)) for k, (a, b) in zip(keys, flags)}


def check_post1(results, post1: dict, expected: Expected) -> None:
    """I9: Layer 1 at the label dates reproduces Post 1's frame counts and quarter flags."""
    frames = [f for r in results for f in r.frames]
    n = {"all_1": sum(f["l1_all_1"] for f in frames),
         "all_100": sum(f["l1_all_100"] for f in frames)}
    n["other"] = len(frames) - n["all_1"] - n["all_100"]
    if n != expected.post1_frames:
        raise Q.CheckFailed("I9", f"frames {n}, Post 1's {expected.post1_frames}")
    ours = {(r.cube, c["date"].isoformat(), c["quarter"]): (c["l1_all_1"], c["l1_all_100"])
            for r in results for c in r.cells}
    bad = [f"{k}: {ours.get(k)} against {post1.get(k)}" for k in sorted(set(ours) | set(post1))
           if ours.get(k) != post1.get(k)]
    if bad:
        raise Q.CheckFailed("I9", bad)
    theirs = {"fully_actual": sum(a for a, _ in post1.values()),
              "entirely_filled": sum(b for _, b in post1.values())}
    if theirs != expected.post1_quarters:
        raise Q.CheckFailed("I9", f"quarters {theirs}, the plan's {expected.post1_quarters}")


def check_same_day_codes(results, expected: Expected) -> None:
    """I10: L2 = 900 or -900 at the label dates, over all pixels, as Post 1's records give."""
    frames = [f for r in results for f in r.frames]
    got = {"frames": sum(1 for f in frames if f["n_l2_900_all"] + f["n_l2_minus900_all"] > 0),
           900: sum(f["n_l2_900_all"] for f in frames),
           -900: sum(f["n_l2_minus900_all"] for f in frames)}
    got["pixels"] = got[900] + got[-900]
    if got != expected.same_day_codes:
        raise Q.CheckFailed("I10", f"{got}, the plan's {expected.same_day_codes}")


def cube_number(cube: str) -> int:
    return int(cube.split("_")[0])


def example_frame(results) -> tuple[str, dict]:
    """I11: the frame with the largest share of its pixels filled and with a future Layer 2
    source; a tie goes to the earliest label date, then the lowest cube number."""
    frames = [(r.cube, f) for r in results for f in r.frames if f["n_l2_future"] > 0]
    if not frames:
        raise Q.CheckFailed("I11", "no frame has a filled pixel with a future Layer 2 source")

    def key(cf):
        cube, f = cf
        return (-Fraction(f["n_l2_future"], f["n_pixels"]), f["date"], cube_number(cube))

    ranked = sorted(frames, key=key)
    if len(ranked) > 1 and key(ranked[0]) == key(ranked[1]):
        raise Q.CheckFailed("I11", "the rule returns more than one frame")
    return ranked[0]


def nth(values: list, k: int, of: int = 10):
    """The inverted-CDF quantile of a list, each value once."""
    if not values:
        return None
    s = sorted(values)
    return s[-(-k * len(s) // of) - 1]


# tables

def share(num, den) -> str:
    return "" if not den else f"{num / den:.6f}"


def fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (dt.date, dt.datetime)):
        return v.isoformat()
    if isinstance(v, Fraction):
        return f"{float(v):.6f}"
    return str(v)


def place_cols(design, cube: str) -> dict:
    p = next(p for p in design.places if p.cube == cube)
    return {"cube": p.cube, "place": p.name}


def daily_rows(design, results) -> list[dict]:
    keys = ["n_pixels", "n_nodata", "n_fully_actual", "n_filled", "n_entirely_filled",
            "n_blended"] + [f"{k}_l3_{s}" for k in ("entirely_filled", "blended")
                            for s, _ in Q.L3_STATUSES]
    return [{**place_cols(design, r.cube), "date": d["date"], "quarter": d["quarter"],
             **{k: d[k] for k in keys}} for r in results for d in r.daily]


def monthly_rows(design, results) -> list[dict]:
    """Estimands 1 and 5 by place and month, and over the two years; both weights."""
    statuses = [s for s, _ in Q.L3_STATUSES] + ["cloud_shadow_haze"]
    out = []
    for r in results:
        classes = dict(zip(design.days, day_classes(r.tile_fully_actual, design)))
        tiles = {}
        for d in r.daily:
            t = tiles.setdefault(d["date"], {})
            for k, v in d.items():
                if k.startswith("n_") or "_l3_" in k:
                    t[k] = t.get(k, 0) + v
        for t in tiles.values():
            for kind in ("entirely_filled", "blended"):
                codes = [t[f"{kind}_l3_code{i}"] for i in range(2, 7)]
                t[f"{kind}_l3_cloud_shadow_haze"] = sum(codes)
        months = sorted({(d.year, d.month) for d in design.days})
        periods = [(f"{y}-{m:02d}", [d for d in design.days if (d.year, d.month) == (y, m)])
                   for y, m in months]
        periods.append((f"{design.days[0]:%Y-%m}..{design.days[-1]:%Y-%m}", list(design.days)))
        for label, days in periods:
            row = {**place_cols(design, r.cube), "period": label, "n_days": len(days)}
            for k in ("n_pixels", "n_nodata", "n_fully_actual", "n_filled",
                      "n_entirely_filled", "n_blended"):
                row["n_pixel_days" if k == "n_pixels" else k] = sum(tiles[d][k] for d in days)
            row["share_fully_actual"] = share(row["n_fully_actual"], row["n_pixel_days"])
            for c in ("all", "some", "none"):
                row[f"days_{c}"] = sum(classes[d] == c for d in days)
            for kind in ("entirely_filled", "blended"):
                with_kind = [d for d in days if tiles[d][f"n_{kind}"] > 0]
                row[f"n_days_with_{kind}"] = len(with_kind)
                for s in statuses:
                    n = sum(tiles[d][f"{kind}_l3_{s}"] for d in days)
                    row[f"n_{kind}_l3_{s}"] = n
                    row[f"share_{kind}_l3_{s}"] = share(n, row[f"n_{kind}"])
                    dw = [Fraction(tiles[d][f"{kind}_l3_{s}"], tiles[d][f"n_{kind}"])
                          for d in with_kind]
                    row[f"share_dw_{kind}_l3_{s}"] = (fmt(sum(dw) / len(dw)) if dw else "")
            out.append(row)
    return out


def gap_tables(design, results) -> tuple[list[dict], list[dict]]:
    """Estimand 2: gaps.csv and gap_lengths.csv. Summaries leave censored gaps out and count
    them apart; pixel gaps are weighted by pixel-days and by gaps, runs of class none by runs."""
    summary, lengths = [], []
    values = np.arange(design.n_days + 1)
    for r in results:
        pc = place_cols(design, r.cube)
        none_days = [n == 0 for n in r.tile_fully_actual]
        levels = (("pixel", r.gaps, ("pixel_days", "gaps")),
                  ("tile_none", Q.runs_of(none_days, design.n_days), ("runs",)))
        for level, hist, weights in levels:
            for kind in Q.CENSORING:
                lengths += [{**pc, "level": level, "censored": kind, "length": length,
                             "n_gaps": int(n), "n_days": int(n) * length}
                            for length, n in enumerate(hist[kind]) if n]
            for weight in weights:
                w = {k: hist[k] * values if weight == "pixel_days" else hist[k]
                     for k in Q.CENSORING}
                summary.append({**pc, "level": level, "weight": weight,
                                "n_uncensored": int(w["none"].sum()),
                                **{f"n_censored_{k}": int(w[k].sum()) for k in Q.CENSORING[1:]},
                                "median": Q.quantile(values, w["none"], 5),
                                "p90": Q.quantile(values, w["none"], 9),
                                "max": Q.largest(values, w["none"])})
    return summary, lengths


def label_row(design, cube, x: dict, quarter=None) -> dict:
    n_filled = x["n_filled"]
    row = {**place_cols(design, cube), "date": x["date"]}
    if quarter is not None:
        row["quarter"] = quarter
    row.update({k: x[k] for k in ("n_pixels", "n_nodata", "n_fully_actual", "n_filled",
                                  "n_entirely_filled", "n_blended")})
    row.update({f"n_l2_{s}": x[f"n_l2_{s}"] for s in Q.SOURCES})
    same = sum(x[f"n_l2_{s}"] for s in Q.SOURCES if s.startswith("same_day"))
    row["share_fully_actual"] = share(x["n_fully_actual"], x["n_pixels"])
    row["share_filled"] = share(n_filled, x["n_pixels"])
    row["share_l2_same_day"] = share(same, n_filled)
    row.update({f"share_l2_{s}": share(x[f"n_l2_{s}"], n_filled) for s in Q.SOURCES})
    row.update(Q.offsets(x["hist"]))
    row["share_future_of_pixels"] = share(x["n_l2_future"], x["n_pixels"])
    row.update({k: x[k] for k in ("l1_all_1", "l1_all_100", "n_l2_900_all",
                                  "n_l2_minus900_all")})
    return row


def label_tables(design, results, example) -> tuple[list, list, list, list]:
    """Estimand 3: label_frames.csv, label_cells.csv, label_codes.csv and label_summary.csv."""
    frames = [label_row(design, r.cube, f) for r in results for f in r.frames]
    cells = [label_row(design, r.cube, c, c["quarter"]) for r in results for c in r.cells]
    codes = [{**place_cols(design, r.cube), **c} for r in results for c in r.codes]
    all_frames = [f for r in results for f in r.frames]
    tot = {k: sum(f[k] for f in all_frames) for k in all_frames[0]
           if k.startswith("n_") and not k.endswith("_all")}
    s = {"n_frames": len(all_frames), **tot}
    with_filled = [f for f in all_frames if f["n_filled"] > 0]
    s["n_frames_with_filled"] = len(with_filled)

    def pooled(name, num, den, frames_dw):
        s[f"share_{name}"] = share(num(tot), den(tot))
        dw = [Fraction(num(f), den(f)) for f in frames_dw]
        s[f"share_dw_{name}"] = fmt(sum(dw) / len(dw)) if dw else ""

    def same(x):
        return sum(x[f"n_l2_{k}"] for k in Q.SOURCES if k.startswith("same_day"))

    pooled("fully_actual", lambda x: x["n_fully_actual"], lambda x: x["n_pixels"], all_frames)
    pooled("filled", lambda x: x["n_filled"], lambda x: x["n_pixels"], all_frames)
    pooled("l2_same_day", same, lambda x: x["n_filled"], with_filled)
    for k in Q.SOURCES:
        pooled(f"l2_{k}", lambda x, k=k: x[f"n_l2_{k}"], lambda x: x["n_filled"], with_filled)
    s.update(Q.offsets(sum(f["hist"] for f in all_frames)))
    cube, f = example
    s.update({"example_cube": cube, "example_place": place_cols(design, cube)["place"],
              "example_date": f["date"], "example_n_l2_future": f["n_l2_future"],
              "example_n_pixels": f["n_pixels"],
              "example_share_future_of_pixels": share(f["n_l2_future"], f["n_pixels"]),
              "median_share_future_of_pixels": fmt(nth(
                  [Fraction(x["n_l2_future"], x["n_pixels"]) for x in all_frames], 5))})
    return frames, cells, codes, [s]


def window_tables(design, results, complete) -> tuple[list, list, list, list]:
    """Estimand 4: windows.csv, window_days.csv, block_coverage.csv and windows_summary.csv."""
    windows, days, blocks = [], [], []
    for r in results:
        pc = place_cols(design, r.cube)
        for w in r.windows:
            row = {**pc, **{k: w.get(k) for k in ("date", "quarter", "window", "days", "first_day",
                                                   "last_day", "complete")}}
            keys = ("n_pixels", "n_covered", "share_covered", "days_p10", "days_p50", "days_p90",
                    "n_days_any")
            row.update(dict.fromkeys(keys))
            if w["complete"]:
                row.update({k: w[k] for k in keys if k != "share_covered"})
                row["share_covered"] = share(w["n_covered"], w["n_pixels"])
            windows.append(row)
        days += [{**pc, **x} for x in r.window_days]
        for b in r.blocks:
            row = {**pc, **b}
            row.update({f"share_blocks_{i}": share(b[f"n_blocks_{i}"], b["n_pixels"])
                        for i in range(4)})
            blocks.append(row)
    for p in design.places:
        found = {w: sorted({x["date"] for x in windows if x["window"] == w and x["complete"]
                            and x["cube"] == p.cube}) for w in complete}
        if found != complete:
            raise Q.CheckFailed("I8", f"{p.cube}: the pass's complete windows aren't those of "
                                      "the plan's geometry")
    common = sorted(set.intersection(*(set(v) for v in complete.values())))
    by_cell = {}
    for x in windows:
        by_cell.setdefault((x["cube"], x["window"], x["date"]), []).append(x)
    hists = {}
    for x in days:
        h = hists.setdefault((x["cube"], x["window"], x["date"]), {})
        h[x["fully_actual_days"]] = h.get(x["fully_actual_days"], 0) + x["n_pixels"]
    summary = []
    for p in design.places:
        for w in design.windows:
            for label, dates in (("own", complete[w.name]), ("common", common)):
                cells = [x for d in dates for x in by_cell[(p.cube, w.name, d)]]
                hist = np.zeros(w.days + 1, np.int64)
                for d in dates:
                    for k, n in hists[(p.cube, w.name, d)].items():
                        hist[k] += n
                values = np.arange(w.days + 1)
                n_pix = sum(x["n_pixels"] for x in cells)
                n_cov = sum(x["n_covered"] for x in cells)
                row = {"cube": p.cube, "place": p.name, "window": w.name, "dates": label,
                       "n_label_dates": len(dates), "n_cells": len(cells), "n_pixels": n_pix,
                       "n_covered": n_cov, "share_covered": share(n_cov, n_pix),
                       "share_covered_median": fmt(nth(
                           [Fraction(x["n_covered"], x["n_pixels"]) for x in cells], 5)),
                       "days_p10": Q.quantile(values, hist, 1),
                       "days_p50": Q.quantile(values, hist, 5),
                       "days_p90": Q.quantile(values, hist, 9),
                       "n_days_any_median": nth([x["n_days_any"] for x in cells], 5)}
                bl = [b for b in blocks if b["cube"] == p.cube and b["date"] in dates]
                is_90 = w.name == "nested_90"
                bp = sum(b["n_pixels"] for b in bl)
                row.update({f"share_blocks_{i}": share(sum(b[f"n_blocks_{i}"] for b in bl), bp)
                            if is_90 else "" for i in range(4)})
                summary.append(row)
    return windows, days, blocks, summary


# expectations

def expectations(design, plan: dict, monthly, gaps, windows) -> list[dict]:
    """EX1-EX7, judged on exact counts before rounding."""
    stmt = {x["id"]: x["statement"] for x in plan["expectations"]}
    two_years = [m for m in monthly if ".." in m["period"]]
    by = {m["cube"]: m for m in two_years}
    ren, an = design.place("Renshou").cube, design.place("Andong").cube
    out = []

    def add(i, observed, met, rule):
        out.append({"id": i, "statement": stmt[i], "observed": observed, "met": bool(met),
                    "rule": rule})

    n_nodata = sum(m["n_nodata"] for m in two_years)
    add("EX1", {"nodata_pixel_days": n_nodata}, n_nodata == 0, "no nodata pixel-day")
    fa = sum(m["n_fully_actual"] for m in two_years)
    total = sum(m["n_pixel_days"] for m in two_years)
    add("EX2", {"fully_actual": fa, "pixel_days": total, "share": share(fa, total)},
        100 * fa >= 15 * total and 100 * fa <= 40 * total, "15% <= share <= 40%, inclusive")
    sh = {c: Fraction(m["n_fully_actual"], m["n_pixel_days"]) for c, m in by.items()}
    others = [v for c, v in sh.items() if c != an]
    add("EX3", {c: fmt(v) for c, v in sh.items()}, all(sh[an] > v for v in others),
        f"{an} strictly higher than each other place")
    some = {c: m["days_some"] for c, m in by.items()}
    add("EX4", {"days_some": some, "days": design.n_days},
        all(3 * v < design.n_days for v in some.values()),
        "3 x days in class some < the days of the series, each place")
    med = {g["cube"]: g["median"] for g in gaps
           if g["level"] == "pixel" and g["weight"] == "pixel_days"}
    others = [v for c, v in med.items() if c != ren]
    add("EX5", {"median_gap_by_pixel_days": med},
        med[ren] is not None and all(v is not None and med[ren] > v for v in others),
        f"{ren} strictly larger than each other place")
    nops = {c: (m["n_entirely_filled_l3_no_planetscope"] + m["n_blended_l3_no_planetscope"],
                m["n_filled"]) for c, m in by.items()}
    frac = {c: Fraction(n, d) if d else None for c, (n, d) in nops.items()}
    ok = all(v is not None for v in frac.values()) and \
        all(frac[ren] > v for c, v in frac.items() if c != ren) and \
        all(frac[an] < v for c, v in frac.items() if c != an)
    add("EX6", {c: {"no_planetscope": n, "filled": d, "share": share(n, d)}
                for c, (n, d) in nops.items()},
        ok, f"{ren} strictly highest and {an} strictly lowest")
    cells = {p.cube: [Fraction(x["n_covered"], x["n_pixels"]) for x in windows
                      if x["cube"] == p.cube and x["window"] == "nested_30" and x["complete"]]
             for p in design.places}
    medians = {c: nth(v, 5) for c, v in cells.items()}
    add("EX7", {"median_covered_share_30_days": {c: fmt(v) for c, v in medians.items()},
                "cells": {c: len(v) for c, v in cells.items()}},
        all(v is not None and 5 * v >= 4 for v in medians.values()),
        "median covered share over the cells >= 80%, each place")
    return out


# writing

def write_csv(path: Path, rows: list[dict]) -> list[list[str]]:
    cols = list(rows[0]) if rows else []
    for r in rows:
        if list(r) != cols:
            raise Q.CheckFailed("tables", f"{path.name}: rows with different columns")
    lines = [cols] + [[fmt(r[c]) for c in cols] for r in rows]
    with open(path, "w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(lines)
    return lines


def read_csv(path: Path) -> list[list[str]]:
    with open(path, newline="") as f:
        return list(csv.reader(f))


def table(lines: list[list[str]], key: tuple, cols: tuple) -> dict:
    head = lines[0]
    ki, ci = [head.index(k) for k in key], [head.index(c) for c in cols]
    return {tuple(r[i] for i in ki): [r[i] for i in ci] for r in lines[1:]}


def check_blocks_sum(windows: list, blocks: list) -> None:
    """In each complete 90-day cell, n1 + 2n2 + 3n3 in block_coverage.csv is the sum of the
    three blocks' covered pixels in windows.csv, and n1 + n2 + n3 the 90-day window's."""
    key = ("cube", "date", "quarter")
    cov = table(windows, key + ("window",), ("n_covered",))
    names = sorted({r[windows[0].index("window")] for r in windows[1:]})
    block_names = [w for w in names if w.startswith("block_")]
    bad = []
    for k, (n0, n1, n2, n3) in table(blocks, key, tuple(f"n_blocks_{i}" for i in range(4))).items():
        n1, n2, n3 = int(n1), int(n2), int(n3)
        per_block = [int(cov[k + (b,)][0]) for b in block_names]
        if n1 + 2 * n2 + 3 * n3 != sum(per_block):
            bad.append(f"{k}: n1 + 2n2 + 3n3 = {n1 + 2 * n2 + 3 * n3}, blocks {per_block}")
        if n1 + n2 + n3 != int(cov[k + ("nested_90",)][0]):
            bad.append(f"{k}: n1 + n2 + n3 isn't the 90-day window's covered pixels")
    if bad:
        raise Q.CheckFailed("blocks_sum", bad)


def check_tables(tmp: Path, written: dict, expected: Expected, design) -> None:
    """The tables, read back, equal what was computed and have the plan's row counts; each label
    cell's counts equal qa_daily.csv's on its day."""
    bad = []
    for name, lines in written.items():
        if read_csv(tmp / name) != lines:
            bad.append(f"{name} reads back differently")
    for name, n in expected.rows.items():
        if len(written[name]) - 1 != n:
            bad.append(f"{name}: {len(written[name]) - 1} rows, the plan says {n}")
    n_win = (len(design.places) * len(design.label_dates) * len(design.quarters)
             * len(design.windows))
    if len(written["windows.csv"]) - 1 != n_win:
        bad.append(f"windows.csv: {len(written['windows.csv']) - 1} rows, not {n_win}")
    daily = read_csv(tmp / "qa_daily.csv")
    cells = read_csv(tmp / "label_cells.csv")
    keys = ["n_pixels", "n_nodata", "n_fully_actual", "n_filled", "n_entirely_filled", "n_blended"]
    di = {c: daily[0].index(c) for c in ["cube", "date", "quarter"] + keys}
    ci = {c: cells[0].index(c) for c in ["cube", "date", "quarter"] + keys}
    day = {(r[di["cube"]], r[di["date"]], r[di["quarter"]]): [r[di[k]] for k in keys]
           for r in daily[1:]}
    for r in cells[1:]:
        if day.get((r[ci["cube"]], r[ci["date"]], r[ci["quarter"]])) != [r[ci[k]] for k in keys]:
            bad.append(f"label_cells.csv {r[:4]}: counts differ from qa_daily.csv")
    if bad:
        raise Q.CheckFailed("tables", bad)
    check_blocks_sum(read_csv(tmp / "windows.csv"), read_csv(tmp / "block_coverage.csv"))


def check_gap_days(results) -> None:
    """Each place's pixel-days in gaps, censored ones included, are its pixel-days not fully
    actual."""
    for r in results:
        in_gaps = sum(int((h * np.arange(h.size)).sum()) for h in r.gaps.values())
        not_fa = sum(d["n_pixels"] - d["n_fully_actual"] for d in r.daily)
        if in_gaps != not_fa:
            raise Q.CheckFailed("gap_days", f"{r.cube}: {in_gaps} pixel-days in gaps, "
                                            f"{not_fa} not fully actual")


def git_state(path: Path = REPO) -> dict:
    try:
        top = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                             capture_output=True, text=True)
        if top.returncode != 0:
            return {"commit": None, "clean": None}
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain",
                                "--untracked-files=no"], capture_output=True, text=True, check=True)
        return {"commit": top.stdout.strip(), "clean": dirty.stdout.strip() == ""}
    except FileNotFoundError:
        return {"commit": None, "clean": None}


def check_commit(state: dict, when: str) -> None:
    """A run from git needs a commit and a clean tree; unknown counts as a failure."""
    if not state.get("commit") or state.get("clean") is not True:
        raise Q.CheckFailed("commit", f"{when}: not a clean git checkout (commit "
                                      f"{state.get('commit')}, clean {state.get('clean')})")


def code_files() -> dict:
    """The SHA-256 of the code that runs, the scripts and the shared module they import."""
    out = {}
    for path in CODE_FILES:
        p = Path(path).resolve()
        name = p.relative_to(REPO.resolve()).as_posix() if p.is_relative_to(REPO.resolve()) \
            else p.name
        out[name] = SP.sha256(p)
    return out


def versions() -> dict:
    out = {"python": sys.version.split()[0], "numpy": np.__version__,
           "rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__}
    for name in ("pandas", "pyarrow"):
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            out[name] = None
    return out


def relative(path: Path, root: Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


# the run

def run(design, record: dict, den_root: Path, out: Path, post1_cells: Path,
        expected: Expected = Expected(), git=None, log=print) -> dict:
    """The plan, start to end. Returns the audit; raises CheckFailed with nothing written."""
    t0 = time.time()
    planet = Path(den_root) / "planet"
    existing = [n for n in OUTPUTS if (Path(out) / n).exists()]
    if existing:
        raise Q.CheckFailed("outputs", f"{out} already holds {existing}; remove them to rerun")
    if git:
        state = git()
        check_commit(state, "before the run")
    else:
        state = {"commit": None, "clean": None, "git": "not used"}
    complete = check_windows(design, expected)
    inv = Q.inventory(design, planet)
    refs, meta = Q.headers(design, inv)
    log(f"inputs: {sum(len(v) for v in inv.qa.values())} PF-QA files; I1, I2 and I8 passed")
    post1 = read_post1_cells(post1_cells)
    results = []
    for p in design.places:
        r = Q.place_pass(design, p, inv.qa[p.cube], refs[p.cube], log=log)
        results.append(r)
        log(f"{p.cube}: {design.n_days} days in {r.seconds:.0f} s")
    check_day_classes(results, design)
    check_post1(results, post1, expected)
    check_same_day_codes(results, expected)
    example = example_frame(results)
    check_gap_days(results)
    sr_path = Q.find_pf_sr(planet, example[0], example[1]["date"])
    sr = Q.read_pf_sr(sr_path, design, refs[example[0]])

    monthly = monthly_rows(design, results)
    gaps, gap_lengths = gap_tables(design, results)
    frames, cells, codes, label_summary = label_tables(design, results, example)
    windows, window_days, blocks, windows_summary = window_tables(design, results, complete)
    tables = {"qa_daily.csv": daily_rows(design, results), "qa_monthly.csv": monthly,
              "gaps.csv": gaps, "gap_lengths.csv": gap_lengths, "label_frames.csv": frames,
              "label_cells.csv": cells, "label_codes.csv": codes,
              "label_summary.csv": label_summary, "windows.csv": windows,
              "window_days.csv": window_days, "block_coverage.csv": blocks,
              "windows_summary.csv": windows_summary}
    ex = expectations(design, record["plan"], monthly, gaps, windows)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f".writing-{os.getpid()}"
    tmp.mkdir()
    try:
        written = {name: write_csv(tmp / name, rows) for name, rows in tables.items()}
        check_tables(tmp, written, expected, design)
        (tmp / "expectations.json").write_text(json.dumps({"expectations": ex}, indent=1) + "\n")
        hashes = [f"{h}  {relative(p, den_root)}" for r in results for h, p in r.hashes]
        hashes.append(f"{sr['sha256']}  {relative(sr_path, den_root)}")
        (tmp / "inputs.sha256").write_text("\n".join(hashes) + "\n")
        if git:
            after = git()
            check_commit(after, "before writing")
            if after["commit"] != state["commit"]:
                raise Q.CheckFailed("commit", "the code changed during the run")
        audit = {
            "plan": {"path": relative(SP.PLAN, REPO), "sha256": record["plan_sha256"],
                     "frozen_sha256": SP.PLAN_SHA256},
            "amendments": {"path": relative(SP.AMENDMENTS, REPO),
                           "sha256": record["amendments_sha256"],
                           "ids": [a["id"] for a in record["amendments"]]},
            "repository_inputs": {relative(post1_cells, REPO) if Path(post1_cells).resolve()
                                  .is_relative_to(REPO.resolve()) else Path(post1_cells).name:
                                  SP.sha256(post1_cells)},
            "code": {**state, "files": code_files()}, "software": versions(),
            "checks": {c: "passed" for c in SP.INVARIANTS + ("pf_sr", "blocks_sum", "gap_days",
                                                             "tables")},
            "inputs": {"pf_qa_files": {p.cube: len(inv.qa[p.cube]) for p in design.places},
                       "other_entries": inv.others,
                       "grids": {c: {"crs": str(ref[0]), "transform": list(ref[1])[:6]}
                                 for c, ref in refs.items()},
                       "metadata": meta,
                       "metadata_beyond_image_structure": sorted(
                           k for k, v in meta.items() if v["beyond_image_structure"])},
            "example_frame": {"cube": example[0], "date": example[1]["date"].isoformat(),
                              "pf_sr": relative(sr_path, den_root), "pf_sr_check": sr},
            "diagnostics": {r.cube: r.diagnostics for r in results},
            "seconds": {**{r.cube: round(r.seconds, 1) for r in results},
                        "total": round(time.time() - t0, 1)},
        }
        (tmp / "audit.json").write_text(json.dumps(audit, indent=1, default=str) + "\n")
        for name in OUTPUTS:
            os.replace(tmp / name, out / name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"every check passed; {len(OUTPUTS)} files written to {out.name}/")
    return audit


def check_inputs(design, record: dict, den_root: Path, out: Path,
                 expected: Expected = Expected(), log=print) -> dict:
    """I1 to I3 on every file and the metadata of each place's first and last files; prints
    and records pass or fail, never a count of codes."""
    t0 = time.time()
    report = {"plan_sha256": record["plan_sha256"], "checks": {}}
    try:
        check_windows(design, expected)
        report["checks"]["I8"] = "passed"
        inv = Q.inventory(design, Path(den_root) / "planet")
        report["checks"]["I1"] = "passed (names)"
        refs, meta = Q.headers(design, inv)
        report["checks"].update({"I1": "passed", "I2": "passed (headers)"})
        report["metadata"] = meta
        report["grids"] = {c: {"crs": str(r[0]), "transform": list(r[1])[:6]}
                           for c, r in refs.items()}
        for p in design.places:
            for day in design.days:
                path = inv.qa[p.cube][day]
                try:
                    with rasterio.open(path) as s:
                        l1, l2, l3 = s.read([1, 2, 3])
                except Q.UNREADABLE as e:
                    raise Q.CheckFailed("I1", f"{p.cube} {path.name}: unreadable ({e})") from e
                bad = Q.code_problems(l1, l2, l3, design.nodata)
                if bad:
                    raise Q.CheckFailed("I3", [f"{p.cube} {path.name}: {x}" for x in bad])
            log(f"{p.cube}: I3 passed on {design.n_days} files, {time.time() - t0:.0f} s")
        report["checks"]["I3"] = "passed"
        report["files"] = {p.cube: len(inv.qa[p.cube]) for p in design.places}
        report["metadata_beyond_image_structure"] = sorted(
            k for k, v in meta.items() if v["beyond_image_structure"])
        report["result"] = "passed"
    except Q.CheckFailed as e:
        report["result"] = f"{e.check} failed"
        report["problems"] = e.problems[:200]
    report["seconds"] = round(time.time() - t0, 1)
    out.mkdir(parents=True, exist_ok=True)
    (out / "input_check.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check-inputs", action="store_true",
                    help="I1 to I3 and the metadata only; computes no estimand")
    ap.add_argument("--without-git", action="store_true",
                    help="for a copy without git: record the code files' SHA-256, not a commit")
    a = ap.parse_args(argv)
    try:
        design, record = SP.load()
    except SP.PlanError as e:
        print(f"plan: {e}")
        return 1
    if a.check_inputs:
        report = check_inputs(design, record, config.DEN_HOME, RUNS / "smoke")
        for k, v in report["checks"].items():
            print(f"{k}: {v}")
        print(f"result: {report['result']} ({report['seconds']} s)")
        for x in report.get("problems", [])[:20]:
            print(f"  {x}")
        if report.get("metadata_beyond_image_structure"):
            print(f"metadata beyond image structure: {report['metadata_beyond_image_structure']}")
        print(f"recorded in {relative(RUNS / 'smoke' / 'input_check.json', REPO)}")
        return 0 if report["result"] == "passed" else 2
    try:
        run(design, record, config.DEN_HOME, RUNS, POST1_CELLS,
            git=None if a.without_git else git_state)
    except Q.CheckFailed as e:
        print(f"stopped: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
