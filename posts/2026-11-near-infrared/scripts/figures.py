#!/usr/bin/env python
"""Figures for the post.

    python posts/2026-11-near-infrared/scripts/figures.py --fig all

Uses the saved checkpoints in runs/models/ and runs/cell_level.csv; every number in a caption
comes from that table. The figure frame (Yangzhou, 2018-04-01, quadrant NW) was chosen after
the results were known; METHOD.md explains how.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

from geoaiwatch import config as C
from geoaiwatch.labels import label_paths, read_label, remap_to_retained

mpl.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "regular",
})

POST = Path(__file__).resolve().parents[1]
ART = POST / "artifacts"
RUNS = POST / "runs"
FIGDIR = POST / "figures"
MODELDIR = RUNS / "models"
SHARES = RUNS / "figure3_shares.json"

CUBE, DATE, FOLD, SEED, EPOCH = "6813_3313_13", "2018-04-01", "NW", 0, 120

# color-blind-safe, ordered as RETAINED_CLASSES
CLASS_COLORS = ["#E69F00",   # agriculture  orange
                "#009E73",   # forest_veg   green
                "#CC79A7",   # impervious   magenta
                "#8C6D31",   # soil         brown
                "#0072B2"]   # water        blue

# display names for the post's figures; data keys are unchanged
DISPLAY = {"agriculture": "farmland", "forest_veg": "forest & other vegetation",
           "impervious": "paved surfaces", "soil": "bare soil", "water": "water"}
PLACES = {"6466": "Renshou", "6730": "Fengcheng", "6813": "Yangzhou", "7026": "Andong"}


def primary_by_fold(Cell: pd.DataFrame, classes) -> pd.DataFrame:
    """Primary estimand: seeds averaged within each cell, cells weighted equally within a fold."""
    e = Cell[Cell.epoch == EPOCH]
    cols = [f"d_{c}" for c in classes]
    per_cell = e.groupby(["fold", "cube", "date"])[cols].mean()
    g = per_cell.groupby("fold").mean()
    g.columns = classes
    return g


ATTRIB = ("Planet Fusion © 2021 Planet Labs Inc. / DynamicEarthNet "
          "(Toker et al. 2022) · CC BY-SA 4.0 · modified")


# io

def load_arm(arm: str):
    import segmentation_models_pytorch as smp
    import torch
    p = MODELDIR / f"{FOLD}_s{SEED}_{arm}_ep{EPOCH}.pt"
    if not p.exists():
        raise SystemExit(f"{p} not found: download it from this post's release (see the README) "
                         f"into {MODELDIR}")
    ck = torch.load(p, map_location="cpu")
    held = {k: ck.get(k) for k in ("arm", "fold", "seed", "epoch")}
    want = {"arm": arm, "fold": FOLD, "seed": SEED, "epoch": EPOCH}
    assert held == want, f"{p.name} holds {held}, expected {want}"
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=4,
                 classes=len(ck["classes"]))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


def load_frame(classes):
    from geoaiwatch.data import read_sr
    d = C.cube_dir(CUBE)
    x = read_sr(d / "PF-SR" / f"{DATE}.tif")                    # RGBN, (4,1024,1024)
    y = remap_to_retained(read_label(label_paths(CUBE)[DATE]), classes)
    return x, y


def quadrant(a):
    """The held-out quadrant, as a view."""
    r0, c0 = C.QUADS[FOLD]
    return a[..., r0:r0 + C.HALF, c0:c0 + C.HALF]


def predict(model, x, mean, std, device):
    import torch
    t = torch.from_numpy(np.array(x, copy=True))
    t = (t - torch.as_tensor(mean).view(4, 1, 1)) / torch.as_tensor(std).view(4, 1, 1)
    if model_arm(model) == "rgb0":
        t[3] = 0.0
    model = model.to(device)
    with torch.no_grad():
        return model(t[None].to(device)).argmax(1)[0].cpu().numpy()


_ARM = {}


def model_arm(m):
    return _ARM[id(m)]


# rendering helpers

def stretch(a, lo=2, hi=98, gamma=1.0):
    """Percentile stretch to [0,1], per channel, with optional gamma."""
    out = np.empty_like(a, dtype=np.float32)
    for i in range(a.shape[0]):
        p1, p2 = np.percentile(a[i], [lo, hi])
        out[i] = np.clip((a[i] - p1) / max(p2 - p1, 1e-9), 0, 1)
    return out ** gamma if gamma != 1.0 else out

def true_color(x):
    # wider percentiles + gamma < 1 lifts the midtones; this landscape is dark
    return stretch(x[[0, 1, 2]], lo=1, hi=99, gamma=0.75).transpose(1, 2, 0)

def false_color(x):
    return stretch(x[[3, 0, 1]]).transpose(1, 2, 0)              # NIR,R,G


def find_ponds(y, water: int, n: int = 3, min_px: int = 150):
    """The n largest compact water bodies in the reference labels (not rivers, not cut by
    the edge): 4-connected components, bounding-box fill >= 0.45, aspect <= 3."""
    m = (y == water)
    seen = np.zeros_like(m, dtype=bool)
    H, W = m.shape
    found = []
    for r0, c0 in zip(*np.nonzero(m)):
        if seen[r0, c0]:
            continue
        stack, pix = [(r0, c0)], []
        seen[r0, c0] = True
        while stack:
            r, c = stack.pop()
            pix.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and m[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        rs = np.array([q[0] for q in pix])
        cs = np.array([q[1] for q in pix])
        h, w = rs.max() - rs.min() + 1, cs.max() - cs.min() + 1
        if (len(pix) >= min_px and len(pix) / (h * w) >= 0.45 and max(h, w) / min(h, w) <= 3
                and rs.min() > 0 and cs.min() > 0 and rs.max() < H - 1 and cs.max() < W - 1):
            found.append((len(pix), rs.mean(), cs.mean(), max(h, w) / 2))
    found.sort(reverse=True)
    assert len(found) >= n, f"only {len(found)} compact ponds in the reference labels"
    return found[:n]


RING = "#FFC400"


def ring_ponds(ax, ponds, dr=0, dc=0):
    for _, r, c, half in ponds:
        ax.add_patch(Circle((c + dc, r + dr), half + 7, fill=False, ec=RING, lw=2.2,
                            path_effects=[pe.Stroke(linewidth=3.8, foreground="black", alpha=0.6),
                                          pe.Normal()]))


def reveal_window(ponds, pad=4):
    """The held-out quadrant, extended where a ring would cross an edge inside the frame."""
    r0, c0 = C.QUADS[FOLD]
    rings = [(r, c, half + 7) for _, r, c, half in ponds]
    top = min(int(np.floor(r - k)) for r, c, k in rings) - pad
    bottom = max(int(np.ceil(r + k)) for r, c, k in rings) + pad + 1
    left = min(int(np.floor(c - k)) for r, c, k in rings) - pad
    right = max(int(np.ceil(c + k)) for r, c, k in rings) + pad + 1
    tile = 2 * C.HALF
    rows = (max(0, r0 + min(0, top)), min(tile, r0 + max(C.HALF, bottom)))
    cols = (max(0, c0 + min(0, left)), min(tile, c0 + max(C.HALF, right)))
    return rows, cols


def tag(ax, s):
    ax.set_title(s)
    ax.set_xticks([])
    ax.set_yticks([])


def legend(fig, classes):
    fig.legend(handles=[Patch(facecolor=c, label=DISPLAY.get(n, n))
                        for n, c in zip(classes, CLASS_COLORS)],
               loc="lower center", ncol=len(classes), frameon=False,
               bbox_to_anchor=(0.5, 0.055))

def footer(fig, text):
    fig.text(0.5, 0.01, text, ha="center", va="bottom", fontsize=6.5,
             color="#555555", wrap=True)

# figures

def fig_reveal(x, cell, yq, classes):
    ponds = find_ponds(yq, classes.index("water"))
    (ra, rb), (ca, cb) = reveal_window(ponds)
    q = x[..., ra:rb, ca:cb]
    r0, c0 = C.QUADS[FOLD]
    shift = {"dr": r0 - ra, "dc": c0 - ca}
    size = f"{cb - ca} × {rb - ra} pixels at 3 m"
    extended = (rb - ra, cb - ca) != (C.HALF, C.HALF)
    fig, ax = plt.subplots(1, 2, figsize=(9, 5.2), layout="constrained")
    ax[0].imshow(true_color(q))
    tag(ax[0], "True color")
    ax[1].imshow(false_color(q))
    tag(ax[1], "Near-infrared shown as red")
    for a_ in ax:
        ring_ponds(a_, ponds, **shift)
    view = "the test quarter, extended to show each ring whole" if extended else "the test quarter"
    footer(fig, f"Yangzhou, Jiangsu · {DATE} · {view}, {size}.\n"
                f"Living plants reflect near-infrared strongly; "
                f"water absorbs almost all of it.\n{ATTRIB}")
    fig.get_layout_engine().set(rect=(0, 0.10, 1, 0.90))
    fig.savefig(FIGDIR / "01_reveal.png")
    plt.close(fig)
    # the post shows the two halves separately
    for name, img, title, note in (
            ("01a_true_color.png", true_color(q), "True color",
             f"Yangzhou, Jiangsu · {DATE} · {size}. "
             f"Circled: three ponds in the reference map.\n{ATTRIB}"),
            ("01b_false_color.png", false_color(q), "Near-infrared shown as red",
             f"Living plants reflect near-infrared strongly; "
             f"water absorbs almost all of it.\n{ATTRIB}")):
        fig, ax = plt.subplots(figsize=(5.2, 5.7), layout="constrained")
        ax.imshow(img)
        tag(ax, title)
        ring_ponds(ax, ponds, **shift)
        footer(fig, note)
        fig.get_layout_engine().set(rect=(0, 0.10, 1, 0.90))
        fig.savefig(FIGDIR / name)
        plt.close(fig)

def fig_predictions(y, p_nir, p_rgb, classes, cell, water_mean):
    cmap = ListedColormap(CLASS_COLORS)
    fig, ax = plt.subplots(1, 3, figsize=(13, 5.4), layout="constrained")
    for a, img, t in zip(
            ax, (y, p_rgb, p_nir),
            ("Reference map", "Color only", "Color + near-infrared")):
        m = np.ma.masked_where(img == C.IGNORE_INDEX, img)
        a.imshow(m, cmap=cmap, vmin=0, vmax=len(classes) - 1,
                 interpolation="nearest")
        tag(a, t)
    legend(fig, classes)
    footer(fig, f"Same model, same training, same random choices; the only difference is "
                f"whether the fourth channel carries near-infrared or zeros.\n"
                f"This test quarter, one training run: mIoU {cell.rgb0_miou:.3f} → "
                f"{cell.rgbn_miou:.3f}; water IoU {cell.rgb0_water:.3f} → {cell.rgbn_water:.3f}. "
                f"Across all 384 test tiles, the water gain averages {water_mean:+.3f}.\n{ATTRIB}")
    fig.get_layout_engine().set(rect=(0, 0.13, 1, 0.87))
    fig.savefig(FIGDIR / "02_predictions.png")
    plt.close(fig)


def fig_disagreement(x, y, p_nir, p_rgb, classes, cell, tol_pct):
    """Where the arms disagree, in both directions."""
    valid = y != C.IGNORE_INDEX
    nir_right = valid & (p_nir == y) & (p_rgb != y)
    rgb_right = valid & (p_rgb == y) & (p_nir != y)

    rgba = np.zeros((*y.shape, 4), np.float32)
    rgba[nir_right] = mpl.colors.to_rgba("#F0E442")              # yellow: in no class key
    rgba[rgb_right] = mpl.colors.to_rgba("#FFFFFF")              # white: in no class key

    q = quadrant(x)
    fig, ax = plt.subplots(figsize=(6.4, 7.2), layout="constrained")
    ax.imshow(true_color(q) * 0.35)
    ax.imshow(rgba)
    tag(ax, "Where only one model is right")

    n_v = int(valid.sum())
    recorded = json.loads(SHARES.read_text())
    for key, got in (("only_near_infrared_right_pct", 100 * nir_right.sum() / n_v),
                     ("only_color_right_pct", 100 * rgb_right.sum() / n_v)):
        assert abs(got - recorded[key]) <= tol_pct, \
            f"Figure 3 {key}: {got:.3f}, recorded {recorded[key]}"
    fig.legend(handles=[
        Patch(facecolor="#F0E442", edgecolor="#777777",
              label=f"Only the near-infrared model right  ({nir_right.sum() / n_v:.0%})"),
        Patch(facecolor="#FFFFFF", edgecolor="#777777",
              label=f"Only the color-only model right  ({rgb_right.sum() / n_v:.0%})")],
        loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, 0.075))
    footer(fig, f"Shares of labeled pixels that one model gets right and the other wrong "
                f"(one training run); their difference is a change in pixel accuracy.\n"
                f"mIoU, a separate class-averaged score, rises "
                f"{cell.rgb0_miou:.3f} → {cell.rgbn_miou:.3f} (+{cell.d_miou:.3f}).  "
                f"{ATTRIB}")
    fig.get_layout_engine().set(rect=(0, 0.17, 1, 0.83))
    fig.savefig(FIGDIR / "03_disagreement.png")
    plt.close(fig)


def fig_perclass(g, classes, thr=0.02):
    """Per-class delta by fold, primary (cell-level) estimand."""
    order = g.mean().sort_values(ascending=False).index.tolist()
    low = order[-1]
    # the caption states these facts; check that the data still support them
    assert low == "impervious", f"smallest mean gain is now {low}"

    fig, ax = plt.subplots(figsize=(7.5, 4.6), layout="constrained")
    xs = np.arange(len(order))
    for i, f in enumerate(g.index):
        ax.scatter(xs + (i - 1.5) * 0.12, g.loc[f, order], s=34,
                   label=f"quarter {f}", zorder=3)
    ax.scatter(xs, g[order].mean(), marker="_", s=900, color="k",
               zorder=4, label="average")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.axhline(0.02, color="#999999", lw=0.6, ls=":")
    ax.axhline(-0.02, color="#999999", lw=0.6, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels([DISPLAY.get(c, c).replace(" & ", " &\n") for c in order])
    ax.set_ylabel("Change in overlap score (IoU) with near-infrared")
    handles, labels = ax.get_legend_handles_labels()
    handles[labels.index("average")] = Line2D([], [], color="k", lw=2)
    ax.legend(handles, labels, frameon=False, ncol=5, fontsize=7.5, loc="upper right")
    ax.set_title("What near-infrared adds, by class")
    footer(fig, f"Each point is one test quarter's average over its 96 test tiles, with the "
                f"three training runs averaged within each tile. Dotted lines mark ±{thr}, the\n"
                f"smallest change counted as meaningful. Paved surfaces had the "
                f"smallest gain ({g[low].mean():+.3f}).")
    fig.get_layout_engine().set(rect=(0, 0.11, 1, 0.89))
    fig.savefig(FIGDIR / "04_per_class.png")
    plt.close(fig)


def fig_perclass_simple(g, classes):
    """The post's version: one bar per class, the average over all test tiles."""
    mean = g.mean().sort_values(ascending=False)
    assert mean.index[0] == "water", "the title says water gained most"
    fig, ax = plt.subplots(figsize=(7.0, 4.4), layout="constrained")
    colors = [CLASS_COLORS[classes.index(c)] for c in mean.index]
    ax.bar(range(len(mean)), mean.values, color=colors, width=0.62)
    for i, v in enumerate(mean.values):
        ax.text(i, v + 0.004, f"{v:+.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_ylim(min(0.0, mean.min()) * 1.2, mean.max() * 1.12)   # room for the value labels
    ax.set_xticks(range(len(mean)))
    ax.set_xticklabels([DISPLAY.get(c, c).replace(" & ", " &\n") for c in mean.index])
    ax.set_ylabel("Change in overlap score (0 to 1)")
    ax.set_title("Water gained far more than any other land type")
    footer(fig, "Average over all 384 test tiles, with the three training runs "
                "averaged within each tile.")
    fig.get_layout_engine().set(rect=(0, 0.06, 1, 0.94))
    fig.savefig(FIGDIR / "04_per_class_simple.png")
    plt.close(fig)


def fig_variability(Cell):
    """Water gain on each date in each place."""
    e = Cell[Cell.epoch == EPOCH]
    g = e.groupby(["cube", "date"], as_index=False).d_water.mean()   # seeds and folds averaged
    cubes = sorted(g.cube.unique())
    assert (g.groupby("cube").size() == 24).all(), "expected one point per date per AOI"
    assert (g.groupby("cube").d_water.mean() > 0).all(), "an AOI's mean water gain is not positive"
    fig, ax = plt.subplots(figsize=(7.5, 4.6), layout="constrained")
    rng = np.random.default_rng(0)
    g["winter"] = g.date.str[5:7].isin(["12", "01", "02", "03"])
    for i, cb in enumerate(cubes):
        gc = g[g.cube == cb].dropna(subset=["d_water"])
        col = f"C{i}"
        jit = rng.uniform(-.14, .14, len(gc))
        w = gc.winter.values
        ax.scatter(np.full(w.sum(), i) + jit[w], gc.d_water.values[w], s=18,
                   facecolors="none", edgecolors=col, lw=1.2, zorder=3)
        ax.scatter(np.full((~w).sum(), i) + jit[~w], gc.d_water.values[~w], s=16,
                   color=col, alpha=.75, zorder=3)
        ax.plot([i - .3, i + .3], [gc.d_water.mean()] * 2, color="k", lw=2, zorder=4)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", color="#555555", label="other dates"),
                       Line2D([], [], ls="", marker="o", mfc="none", mec="#555555",
                                        label="December to March")],
              frameon=False, fontsize=8, loc="upper left")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xticks(range(len(cubes)))
    ax.set_xticklabels([PLACES.get(c.split("_")[0], c.split("_")[0]) for c in cubes])
    ax.set_ylabel("Gain in water overlap score with near-infrared")
    ax.set_title("Water: the gain holds in all four places, but not evenly")
    footer(fig, "One point per date in each place, averaged over the three training runs and "
                "the four test quarters; open circles are December to March.\nBars are each "
                "place's average, over dates where the gain is large and dates where it is not.")
    fig.get_layout_engine().set(rect=(0, 0.11, 1, 0.89))
    fig.savefig(FIGDIR / "05_variability.png")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fig", choices=["reveal", "predictions", "disagreement",
                                     "per_class", "variability", "all"],
                   default="all")
    p.add_argument("--tolerance", type=float, default=1e-6,
                   help="largest allowed gap from the committed mIoU; CPUs and other GPUs "
                        "differ slightly from the runs")
    a = p.parse_args()

    FIGDIR.mkdir(parents=True, exist_ok=True)
    classes = C.frozen(ART)["retained_classes"]

    Cell = pd.read_csv(RUNS / "cell_level.csv")
    prim = primary_by_fold(Cell, classes)
    sel = Cell[(Cell.cube == CUBE) & (Cell.date.str[:10] == DATE)
               & (Cell.fold == FOLD) & (Cell.seed == SEED)
               & (Cell.epoch == EPOCH)]
    assert len(sel) == 1, f"expected 1 cell row, got {len(sel)}"
    cell = sel.iloc[0]
    print(f"{CUBE} {DATE} fold {FOLD} seed {SEED} epoch {EPOCH}")
    print(f"  d_miou {cell.d_miou:+.4f}   d_water {cell.d_water:+.4f}")

    needs_model = a.fig in ("predictions", "disagreement", "all")
    if needs_model:
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("figure device:", dev)
        x, y_full = load_frame(classes)
        preds = {}
        for arm in ("rgbn", "rgb0"):
            m, ck = load_arm(arm)
            _ARM[id(m)] = arm
            preds[arm] = predict(m, x, ck["norm_mean"], ck["norm_std"], dev)
            del m

        yq = quadrant(y_full)
        pq = {k: quadrant(v) for k, v in preds.items()}

        # check that the rendered panels reproduce the committed IoU
        from train import confusion, metrics
        for arm in ("rgbn", "rgb0"):
            cm = confusion(torch.from_numpy(pq[arm]), torch.from_numpy(yq), len(classes))
            got, want = metrics(cm, classes)["miou_strict"], cell[f"{arm}_miou"]
            print(f"  {arm}: mIoU {got:.6f}, committed {want:.6f}, diff {abs(got - want):.2e}")
            assert abs(got - want) < a.tolerance, (arm, got, want)


    if a.fig in ("reveal", "all"):
        if not needs_model:
            x, y_full = load_frame(classes)
            yq = quadrant(y_full)
        fig_reveal(x, cell, yq, classes)
    if a.fig in ("predictions", "all"):
        fig_predictions(yq, pq["rgbn"], pq["rgb0"], classes, cell, prim["water"].mean())
    if a.fig in ("disagreement", "all"):
        fig_disagreement(x, yq, pq["rgbn"], pq["rgb0"], classes, cell, 0.05 + 100 * a.tolerance)
    if a.fig in ("per_class", "all"):
        fig_perclass(prim, classes)
        fig_perclass_simple(prim, classes)
    if a.fig in ("variability", "all"):
        fig_variability(Cell)

    print(f"\nwrote {FIGDIR}")
    for f in sorted(FIGDIR.glob("*.png")):
        print(f"  {f.name}  {f.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
