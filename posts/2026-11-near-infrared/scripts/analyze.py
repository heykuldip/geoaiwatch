#!/usr/bin/env python
"""Analyze the RGB vs RGB+NIR grid from runs/grid/*.json; no model is re-run.

    python posts/2026-11-near-infrared/scripts/analyze.py                    # all sections
    python posts/2026-11-near-infrared/scripts/analyze.py --section primary  # one section
    python posts/2026-11-near-infrared/scripts/analyze.py --csv              # also tidy CSVs

The primary estimand is cell-level: seeds averaged within each test cell (place x date x
held-out quadrant), cells weighted equally within a fold. The pooled confusion matrix per fold
is a sensitivity analysis. METHOD.md describes both and what the intervals mean.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

POST = Path(__file__).resolve().parents[1]
ART = POST / "artifacts"
RUNS = POST / "runs"

ARMS = ("rgbn", "rgb0")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)


# load

FOLDS, SEEDS = ("NE", "NW", "SE", "SW"), (0, 1, 2)
DESIGN = {"n_frames": 96, "epochs": 120, "eval_epochs": [20, 40, 80, 120],
          "primary_eval_epoch": 120, "batch_size": 16, "crop": 256, "crops_per_frame": 8,
          "lr": 1e-4, "wd": 1e-4}
BATCHES_PER_EPOCH = 96 * 8 // 16
EPOCHS = ("20", "40", "80", "120")
DATES = [f"{y}-{m:02d}-01" for y in (2018, 2019) for m in range(1, 13)]


def frozen_cells() -> set[tuple[str, str]]:
    """The 96 (place, date) cells of the frozen design."""
    aois = json.loads((ART / "screen_result.json").read_text())["eligible_aois"]
    return {(re.match(r"\d+_\d+_\d+", a).group(0), d) for a in aois for d in DATES}


def check_design(runs: list[dict], audit: dict | None = None) -> None:
    """Each fold and seed exactly once, on the frozen settings, complete and uncapped.

    Runs written before batch counts were recorded pass only with an entry in the folder's
    audit.json, which ties each one to its training log.
    """
    cells = frozen_cells()
    pairs = sorted((r["fold"], r["seed"]) for r in runs)
    want = sorted((f, s) for f in FOLDS for s in SEEDS)
    if pairs != want:
        raise RuntimeError(f"expected each of the {len(want)} fold-seed pairs once, found {pairs}")
    for r in runs:
        name = f"{r['fold']}_s{r['seed']}"
        off = {k: r.get(k) for k, v in DESIGN.items() if r.get(k) != v}
        if off:
            raise RuntimeError(f"{name} departs from the frozen design: {off}")
        held = sorted(r.get("checkpoints", {}), key=int)
        if held != list(EPOCHS):
            raise RuntimeError(f"{name}: checkpoints {held}, expected {list(EPOCHS)}")
        for ep in EPOCHS:
            for arm in ("rgbn", "rgb0"):
                pc = r["checkpoints"][ep]["arms"][arm]["per_cell"]
                if len(pc) != len(cells) or {(c["cube"], c["date"]) for c in pc} != cells:
                    raise RuntimeError(
                        f"{name}: epoch {ep}, {arm}: not the {len(cells)} frozen cells")
        hist = r.get("history", [])
        if [h.get("epoch") for h in hist] != list(range(1, DESIGN["epochs"] + 1)):
            raise RuntimeError(
                f"{name}: history doesn't hold epochs 1 to {DESIGN['epochs']} once each")
        if r.get("max_batches") or r.get("max_frames"):
            raise RuntimeError(f"{name} was capped")
        if all("batches" in h for h in hist) and "max_batches" in r and "max_frames" in r:
            counts = {h["batches"] for h in hist}
        else:
            entry = (audit or {}).get(name)
            if entry is None:
                raise RuntimeError(
                    f"{name} has no recorded batch counts and no entry in audit.json")
            logs, sums = entry.get("logs") or [], entry.get("sha256") or []
            if not logs or len(logs) != len(sums) or \
                    not all(re.fullmatch(r"[0-9a-f]{64}", str(h)) for h in sums):
                raise RuntimeError(
                    f"{name}: its audit entry doesn't name its logs with their SHA-256")
            if entry.get("epochs") != DESIGN["epochs"]:
                raise RuntimeError(f"{name}: its log shows {entry.get('epochs')} epochs")
            counts = set(entry["batches"])
        if counts != {BATCHES_PER_EPOCH}:
            raise RuntimeError(f"{name}: batches per epoch {sorted(counts)}, "
                               f"expected {BATCHES_PER_EPOCH}")


def load_runs(d: Path) -> list[dict]:
    files = sorted(p for p in d.glob("*.json") if p.name != "audit.json")
    if not files:
        raise FileNotFoundError(f"no runs in {d}")
    runs = [json.loads(f.read_text()) for f in files]

    smoke = [r for r in runs if r.get("smoke")]
    if smoke:
        raise RuntimeError(f"{len(smoke)} smoke run(s) in {d} — "
                           "smoke output is diagnostic, not analysis")
    audit = d / "audit.json"
    check_design(runs, json.loads(audit.read_text())["runs"] if audit.exists() else None)

    # check that every run shares the frozen design
    ref = runs[0]
    for r in runs[1:]:
        for key in ("classes", "epochs", "eval_epochs", "primary_eval_epoch",
                    "batch_size", "crop", "crops_per_frame", "lr", "wd",
                    "n_frames"):
            assert r[key] == ref[key], (
                f"{r['fold']}_s{r['seed']}: {key} differs from "
                f"{ref['fold']}_s{ref['seed']}")
    print(f"{len(runs)} runs   folds "
          f"{sorted({r['fold'] for r in runs})}   "
          f"seeds {sorted({r['seed'] for r in runs})}")
    print(f"frozen: {ref['n_frames']} frames, {ref['epochs']} epochs, "
          f"batch {ref['batch_size']}, crop {ref['crop']}, "
          f"primary epoch {ref['primary_eval_epoch']}")
    return runs


def tidy_cells(runs: list[dict]) -> pd.DataFrame:
    """One row per (fold, seed, epoch, cube, date): paired deltas + per-arm IoU."""
    rows = []
    for r in runs:
        classes = r["classes"]
        for ep, ck in r["checkpoints"].items():
            zn = {(c["cube"], c["date"]): c for c in ck["arms"]["rgbn"]["per_cell"]}
            zz = {(c["cube"], c["date"]): c for c in ck["arms"]["rgb0"]["per_cell"]}
            for key in sorted(zn):
                cube, date = key
                row = {"fold": r["fold"], "seed": r["seed"], "epoch": int(ep),
                       "cube": cube, "date": date}
                for c in classes:
                    a, b = zn[key]["iou"][c], zz[key]["iou"][c]
                    row[f"rgbn_{c}"] = a
                    row[f"rgb0_{c}"] = b
                    row[f"d_{c}"] = (None if a is None or b is None else a - b)
                    row[f"sup_{c}"] = zn[key]["support"][c]
                row["rgbn_miou"] = zn[key]["miou_strict"]
                row["rgb0_miou"] = zz[key]["miou_strict"]
                row["d_miou"] = (None if row["rgbn_miou"] is None
                                 or row["rgb0_miou"] is None
                                 else row["rgbn_miou"] - row["rgb0_miou"])
                for arm, z in (("rgbn", zn), ("rgb0", zz)):
                    for k2, v in z[key]["confusion_rate"].items():
                        row[f"{arm}_{k2}"] = v
                rows.append(row)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def tidy_folds(runs: list[dict]) -> pd.DataFrame:
    """One row per (fold, seed, epoch): fold-aggregate metrics."""
    rows = []
    for r in runs:
        for ep, ck in r["checkpoints"].items():
            row = {"fold": r["fold"], "seed": r["seed"], "epoch": int(ep),
                   "d_miou": ck["delta_miou_strict"],
                   "rgbn_miou": ck["arms"]["rgbn"]["miou_strict"],
                   "rgb0_miou": ck["arms"]["rgb0"]["miou_strict"],
                   "train_loss_rgbn": ck["train_loss"]["rgbn"],
                   "train_loss_rgb0": ck["train_loss"]["rgb0"]}
            for c in r["classes"]:
                row[f"d_{c}"] = ck["delta_iou"][c]
                row[f"rgbn_{c}"] = ck["arms"]["rgbn"]["iou"][c]
                row[f"rgb0_{c}"] = ck["arms"]["rgb0"]["iou"][c]
            for arm in ARMS:
                for k2, v in ck["arms"][arm]["confusion_rate"].items():
                    row[f"{arm}_{k2}"] = v
            rows.append(row)
    return pd.DataFrame(rows)


# the primary estimand

KEYS = ("fold", "seed", "epoch", "cube", "date")


def cell_means(Cell: pd.DataFrame) -> pd.DataFrame:
    """Seeds averaged within each test cell: one row per (epoch, fold, cube, date)."""
    num = [c for c in Cell.columns if c not in KEYS]
    n_ep = Cell.epoch.nunique()
    per_run = Cell.groupby(["fold", "seed", "epoch"]).size()
    assert len(per_run) == len(FOLDS) * len(SEEDS) * n_ep, \
        f"{len(per_run)} run-epochs for {n_ep} epoch(s)"
    assert (per_run == 96).all(), f"runs without 96 cells: {per_run[per_run != 96].to_dict()}"
    g = Cell.groupby(["epoch", "fold", "cube", "date"])
    want = n_ep * len(FOLDS) * len(frozen_cells())
    assert g.ngroups == want, f"{g.ngroups} cell-epoch groups, expected {want}"
    seeds = g.seed.apply(sorted)
    bad = seeds[seeds.apply(lambda s: s != list(SEEDS))]
    assert bad.empty, f"{len(bad)} cells without seeds 0, 1 and 2 once each"
    return g[num].mean().reset_index()


def fold_from_cells(CM: pd.DataFrame) -> pd.DataFrame:
    """Cells weighted equally within a fold: one row per (epoch, fold)."""
    num = [c for c in CM.columns if c not in ("epoch", "fold", "cube", "date")]
    return CM.groupby(["epoch", "fold"])[num].mean().reset_index()


def check_support(Cell: pd.DataFrame, runs: list[dict]) -> None:
    """Check that every test cell holds >= min_test_support_pixels of every retained class."""
    ep, classes = runs[0]["primary_eval_epoch"], runs[0]["classes"]
    floor = runs[0]["min_test_support_pixels"]
    e = Cell[Cell.epoch == ep]
    cols = [f"sup_{c}" for c in classes]
    assert (e.groupby(["fold", "cube", "date"])[cols].nunique() == 1).all().all(), \
        "support differs between seeds of one cell"
    sup = e.drop_duplicates(["fold", "cube", "date"])[cols]
    low = int((sup < floor).values.sum())
    assert low == 0, f"{low} cell x class pairs below {floor} pixels"
    print(f"support invariant: all {sup.size:,} cell x class pairs >= {floor} px "
          f"(smallest {int(sup.min().min()):,})")


def by_fold_table(g: pd.DataFrame, classes, thr) -> None:
    v = g.set_index("fold")["d_miou"]
    print(f"\ndelta mIoU across {len(v)} folds: "
          f"mean {v.mean():+.4f}  median {v.median():+.4f}  "
          f"min {v.min():+.4f}  max {v.max():+.4f}  sd {v.std(ddof=1):.4f}")
    print(f"folds where RGB+NIR is better: {int((v > 0).sum())}/{len(v)}")
    print(f"\nper-class delta IoU by fold (* = |d| >= {thr}):")
    pc = g.set_index("fold")[[f"d_{c}" for c in classes]]
    pc.columns = classes
    out = pc.round(4).astype(object)
    for c in classes:
        out[c] = [f"{x:+.4f}{'*' if abs(x) >= thr else ''}" for x in pc[c]]
    print(out.to_string())
    print("\nacross folds, per class:")
    s = pd.DataFrame({"mean": pc.mean(), "median": pc.median(),
                      "min": pc.min(), "max": pc.max(), "sd": pc.std(ddof=1),
                      "folds_positive": (pc > 0).sum(),
                      "folds_beyond_threshold": (pc.abs() >= thr).sum()})
    print(s.round(4).to_string())


# sections

def sec_primary(Cell: pd.DataFrame, CF: pd.DataFrame, runs: list[dict]) -> None:
    ep = runs[0]["primary_eval_epoch"]
    classes = runs[0]["classes"]
    thr = runs[0]["min_meaningful_delta_iou"]
    g = CF[CF.epoch == ep]
    print(f"\n{'=' * 78}\nPRIMARY RESULT, cell-level — epoch {ep}\n{'=' * 78}")
    print("Seeds averaged within each test cell; cells weighted equally within a fold.")
    print("\nfold-level:")
    print(g.set_index("fold")[["d_miou", "rgbn_miou", "rgb0_miou"]].round(4).to_string())
    by_fold_table(g, classes, thr)

    e = Cell[Cell.epoch == ep]
    seed_fold = e.groupby(["fold", "seed"]).d_miou.mean()
    print(f"\nwithin-fold seed sd of the fold's cell-mean delta mIoU: "
          f"{seed_fold.groupby('fold').std(ddof=1).round(4).to_dict()}")
    print("  (optimization variance; the fold-level sd above is the spatial term)")
    cm = cell_means(e)
    print(f"cells with RGB+NIR better: {int((cm.d_miou > 0).sum())}/{len(cm)} "
          f"({(cm.d_miou > 0).mean():.1%})")


def sec_pooled(F: pd.DataFrame, runs: list[dict]) -> None:
    ep = runs[0]["primary_eval_epoch"]
    classes = runs[0]["classes"]
    thr = runs[0]["min_meaningful_delta_iou"]
    g = F[F.epoch == ep].groupby("fold").mean(numeric_only=True).reset_index()
    print(f"\n{'=' * 78}\nSENSITIVITY: pooled confusion, pixel-weighted "
          f"— epoch {ep}\n{'=' * 78}")
    print("One confusion matrix per fold and seed, summed over 96 cells; seeds "
          "averaged within fold.")
    print("\nfold-level:")
    print(g.set_index("fold")[["d_miou", "rgbn_miou", "rgb0_miou"]].round(4).to_string())
    by_fold_table(g, classes, thr)


def sec_checkpoints(F: pd.DataFrame, CF: pd.DataFrame, runs: list[dict]) -> None:
    classes = runs[0]["classes"]
    print(f"\n{'=' * 78}\nCHECKPOINT SENSITIVITY\n{'=' * 78}")
    print("The convergence probe showed the training-loss gap narrowing with "
          "training.\nWhether the held-out delta narrows likewise is the "
          "question these checkpoints answer.")
    held = CF.groupby("epoch").agg(rgbn=("rgbn_miou", "mean"), rgb0=("rgb0_miou", "mean"),
                                   d_miou=("d_miou", "mean"))
    t = F.groupby("epoch").agg(d_miou_pooled=("d_miou", "mean"),
                               tl_rgbn=("train_loss_rgbn", "mean"),
                               tl_rgb0=("train_loss_rgb0", "mean"))
    t["train_gap"] = t.tl_rgb0 - t.tl_rgbn
    print("\nheld-out: cell-level primary (means over 4 folds); training loss: "
          "means over 4 folds x 3 seeds")
    print(held.join(t).round(4).to_string())
    print("\ncell-level delta mIoU by fold and epoch:")
    print(CF.pivot_table(index="fold", columns="epoch", values="d_miou").round(4).to_string())
    print("\ncell-level per-class delta by epoch (mean over folds):")
    print(CF.groupby("epoch")[[f"d_{c}" for c in classes]]
          .mean().rename(columns=lambda s: s[2:]).round(4).to_string())


def sec_diagnostics(CF: pd.DataFrame, runs: list[dict]) -> None:
    ep = runs[0]["primary_eval_epoch"]
    g = CF[CF.epoch == ep]
    print(f"\n{'=' * 78}\nDIAGNOSTICS, cell-level — epoch {ep}\n{'=' * 78}")
    print("Expectation in train.py: NIR should help vegetation vs non-vegetated\n"
          "but not necessarily agriculture vs forest, since both are vegetated and a\n"
          "single date carries no phenology.")
    cols = [c for c in g.columns if "_to_" in c]
    print("\nconfusion rates (fraction of truth class predicted as the other):")
    print(g.set_index("fold")[cols].round(4).to_string())
    print("\nmean over folds:")
    m = g[cols].mean()
    for pair in ("agriculture_to_forest_veg", "forest_veg_to_agriculture"):
        a, b = m.get(f"rgbn_{pair}"), m.get(f"rgb0_{pair}")
        if a is not None and b is not None:
            print(f"  {pair:32s} rgbn {a:.4f}  rgb0 {b:.4f}  delta {a - b:+.4f}")
    print("  (negative delta = RGB+NIR confuses the pair less)")


def sec_qa(Cell: pd.DataFrame, runs: list[dict]) -> None:
    ep = runs[0]["primary_eval_epoch"]
    classes = runs[0]["classes"]
    print(f"\n{'=' * 78}\nQA-STATUS HETEROGENEITY, cell-level — epoch {ep}"
          f"\n{'=' * 78}")
    qa = pd.read_parquet(ART / "cell_qa_status.parquet")
    qa["date"] = pd.to_datetime(qa["date"])
    g = cell_means(Cell[Cell.epoch == ep])
    # the held-out quadrant of a fold is that fold's name
    g = g.merge(qa, left_on=["cube", "date", "fold"], right_on=["cube", "date", "quad"],
                how="left", validate="one_to_one")
    miss = int(g.mean_layer1_gapfill_code.isna().sum())
    assert miss == 0, f"{miss} cells did not join to QA status"
    n_obs, n_syn = int(g.fully_observed_cell.sum()), int(g.fully_synthetic_cell.sum())
    print(f"\ncells at epoch {ep}: {len(g)}, seeds averaged within each "
          f"(fully actual {n_obs}, fully synthetic {n_syn}, other {len(g) - n_obs - n_syn})")
    print("Descriptive effect heterogeneity, not a causal estimate of gap\n"
          "filling: QA status covaries with AOI, date and season.")

    def block(mask, name):
        z = g[mask]
        print(f"\n{name}  (n={len(z)} cells)")
        row = {"d_miou": z.d_miou.mean(), **{c: z[f"d_{c}"].mean() for c in classes}}
        print("  " + "  ".join(f"{k} {v:+.4f}" for k, v in row.items()))

    block(g.fully_observed_cell, "FULLY ACTUAL (Layer 1 == 1 everywhere)")
    block(g.fully_synthetic_cell, "FULLY SYNTHETIC (Layer 1 == 100 everywhere)")
    r = np.corrcoef(g.mean_normalized_gapfill_score, g.d_miou)[0, 1]
    print(f"\ndelta mIoU vs mean_normalized_gapfill_score: r = {r:+.3f} (n = {len(g)} cells)")
    print("  (a 0-1 rescaling of the historical Layer-1 code, not an "
          "asserted physical fraction)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=RUNS / "grid_isolated")
    p.add_argument("--section", choices=["primary", "pooled", "checkpoints",
                                         "diagnostics", "qa", "all"],
                   default="all")
    p.add_argument("--csv", action="store_true",
                   help="write tidy CSVs, by default beside the runs folder")
    p.add_argument("--out", type=Path, default=None, help="folder for the --csv tables")
    a = p.parse_args()

    runs = load_runs(a.runs)
    Fold = tidy_folds(runs)
    Cell = tidy_cells(runs)
    CF = fold_from_cells(cell_means(Cell))

    if a.csv:
        out = a.out or a.runs.parent
        out.mkdir(parents=True, exist_ok=True)
        Fold.to_csv(out / "fold_level.csv", index=False)
        Cell.to_csv(out / "cell_level.csv", index=False)
        print(f"wrote {out}/fold_level.csv, cell_level.csv", file=sys.stderr)

    print("Primary estimand: cell-level. Pooled confusion: sensitivity.")
    check_support(Cell, runs)
    if a.section in ("primary", "all"):
        sec_primary(Cell, CF, runs)
    if a.section in ("pooled", "all"):
        sec_pooled(Fold, runs)
    if a.section in ("checkpoints", "all"):
        sec_checkpoints(Fold, CF, runs)
    if a.section in ("diagnostics", "all"):
        sec_diagnostics(CF, runs)
    if a.section in ("qa", "all"):
        sec_qa(Cell, runs)

    print(f"\n{'=' * 78}")
    print("SCOPE: four support-selected AOIs in East Asia, one architecture,\n"
          "one task, four spatial folds sharing most of their training pixels.\n"
          "Not a statement about DynamicEarthNet generally.")


if __name__ == "__main__":
    main()
