#!/usr/bin/env python3
"""Statistics for the figure frames, from the post's runs/ and artifacts/ tables.

For each frame: both arms' three-seed mean IoU at epoch 120, each seed's gain, the frame's rank
among the 384 test cells, its label support and its QA status.

    python posts/2026-11-near-infrared/scripts/frame_stats.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

POST = Path(__file__).resolve().parents[1]
EPOCH = 120
CLASSES = ["agriculture", "forest_veg", "impervious", "soil", "water"]
FRAMES = [
    ("Yangzhou (Post 1 figure)", "6813_3313_13", "2018-04-01", "NW"),
    ("Andong (the rule's pick)", "7026_3201_13", "2019-11-01", "NW"),
]


def main(tables: Path = POST / "runs") -> None:
    c = pd.read_csv(tables / "cell_level.csv")
    c = c[c.epoch == EPOCH]
    cell = c.groupby(["cube", "date", "fold"]).mean(numeric_only=True).reset_index()
    assert len(cell) == 384, f"expected 384 cells at epoch {EPOCH}, found {len(cell)}"
    qa = pd.read_parquet(POST / "artifacts" / "cell_qa_status.parquet")
    print(f"epoch {EPOCH}; 384 cells, seeds averaged within each cell")
    print(f"all cells: ΔmIoU mean {cell.d_miou.mean():+.3f}, median {cell.d_miou.median():+.3f}; "
          f"Δwater mean {cell.d_water.mean():+.3f}, median {cell.d_water.median():+.3f}")
    for name, cube, date, quad in FRAMES:
        s = c[(c.cube == cube) & (c.date == date) & (c.fold == quad)].sort_values("seed")
        assert len(s) == 3, f"{name}: expected 3 seeds, found {len(s)}"
        m = cell[(cell.cube == cube) & (cell.date == date) & (cell.fold == quad)].iloc[0]
        sup = {k: int(m[f"sup_{k}"]) for k in CLASSES}
        tot = sum(sup.values())
        q = qa[(qa.cube == cube) & (qa.date == date) & (qa.quad == quad)].iloc[0]
        print(f"\n=== {name}: {cube} {date} {quad}")
        print(f"  mIoU   RGB {m.rgb0_miou:.3f} -> RGB+NIR {m.rgbn_miou:.3f}  Δ {m.d_miou:+.3f} "
              f"(seeds {', '.join(f'{v:+.3f}' for v in s.d_miou)}); "
              f"{(cell.d_miou < m.d_miou).mean():.0%} of cells gain less; "
              f"{m.d_miou / cell.d_miou.median():.1f}x the median")
        print(f"  water  RGB {m.rgb0_water:.3f} -> RGB+NIR {m.rgbn_water:.3f}  Δ {m.d_water:+.3f} "
              f"(seeds {', '.join(f'{v:+.3f}' for v in s.d_water)}); "
              f"{(cell.d_water < m.d_water).mean():.0%} of cells gain less")
        print("  per-class Δ: " + ", ".join(f"{k} {m[f'd_{k}']:+.3f}" for k in CLASSES))
        print("  land cover: " + ", ".join(f"{k} {sup[k] / tot:.1%}" for k in CLASSES))
        print(f"  QA: fully observed pixels {q.fully_observed_pixel_fraction:.1%}, "
              f"fully synthetic pixels {q.fully_synthetic_pixel_fraction:.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=Path, default=POST / "runs",
                    help="folder holding cell_level.csv")
    main(ap.parse_args().tables)
