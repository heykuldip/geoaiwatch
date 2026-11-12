"""Each number in the November 2026 post, recomputed from the post's results tables.

"A colorful illusion: the invisible color that finds water". The primary tables come from the
runs with training crops inside the training squares; the earlier runs, whose crops overlapped
the held-out quarter, are kept as a sensitivity analysis in runs/overlapping/. Figure 3's pixel
shares need the released checkpoints, so figures.py checks those when it draws the figure.
"""
from pathlib import Path

import pandas as pd
import pytest

POST = Path(__file__).resolve().parents[1] / "posts" / "2026-11-near-infrared"
CLASSES = ["agriculture", "forest_veg", "impervious", "soil", "water"]
KEY = ["fold", "cube", "date"]
WINTER = ["12", "01", "02", "03"]


@pytest.fixture(scope="module")
def runs():
    return pd.read_csv(POST / "runs" / "cell_level.csv")


def cells(runs, epoch=120):
    """Seeds averaged within each test cell (fold x place x date)."""
    return runs[runs.epoch == epoch].groupby(KEY).mean(numeric_only=True).reset_index()


def by_fold(df, col):
    return df.groupby("fold")[col].mean()


def signed(x):
    return f"{x:+.3f}"


def test_overall_score(runs):
    k = cells(runs)
    gain = by_fold(k, "d_miou")
    assert f"{by_fold(k, 'rgb0_miou').mean():.3f}" == "0.337"
    assert f"{by_fold(k, 'rgbn_miou').mean():.3f}" == "0.438"
    assert signed(gain.mean()) == "+0.101"
    assert (gain > 0).all(), "It went up in each of the four test quarters"
    assert ((k.d_miou > 0).sum(), len(k)) == (382, 384)


def test_gains_by_class(runs):
    k = cells(runs)
    gain = {c: by_fold(k, f"d_{c}") for c in CLASSES}
    assert {c: signed(g.mean()) for c, g in gain.items()} == {
        "water": "+0.382", "forest_veg": "+0.047", "soil": "+0.024",
        "agriculture": "+0.040", "impervious": "+0.009"}
    share = gain["water"].mean() / (len(CLASSES) * by_fold(k, "d_miou").mean())
    assert 0.70 < share < 0.80, "about three-quarters comes from water"
    assert (gain["water"] > 0.3).all(), "Water's gain was large in every test quarter"
    for c in ("agriculture", "forest_veg", "soil"):
        assert (gain[c] >= 0.02).all(), f"{c}: small but steady, at least +0.02 in every quarter"
    assert min(gain, key=lambda c: gain[c].mean()) == "impervious"
    assert {c: int((g >= 0.02).sum()) for c, g in gain.items()} == {
        "water": 4, "forest_veg": 4, "soil": 4, "agriculture": 4, "impervious": 1}


def test_confusion_between_farmland_and_forest(runs):
    k = cells(runs)

    def change(a, b):
        return by_fold(k, f"rgbn_{a}_to_{b}").mean() - by_fold(k, f"rgb0_{a}_to_{b}").mean()

    assert signed(change("agriculture", "forest_veg")) == "-0.017", \
        "less likely to mistake farmland for forest"
    assert signed(change("forest_veg", "agriculture")) == "+0.003"


def test_checkpoints_pooling_and_support(runs):
    at = {ep: signed(by_fold(cells(runs, ep), "d_miou").mean()) for ep in (20, 40, 80, 120)}
    assert at == {20: "+0.100", 40: "+0.099", 80: "+0.097", 120: "+0.101"}
    pooled = pd.read_csv(POST / "runs" / "fold_level.csv")
    assert signed(pooled[pooled.epoch == 120].groupby("fold").d_miou.mean().mean()) == "+0.110"
    k = cells(runs)
    support = {c: int(k[f"sup_{c}"].min()) for c in CLASSES}
    assert min(support.values()) >= 1024 and support["water"] == 1674


def test_water_by_place_and_date(runs):
    d = runs[runs.epoch == 120].groupby(["cube", "date"]).d_water.mean().reset_index()
    place = {cube[:4]: signed(v) for cube, v in d.groupby("cube").d_water.mean().items()}
    assert place == {"6466": "+0.194", "6730": "+0.399", "6813": "+0.432", "7026": "+0.504"}
    spread = d.groupby("cube").d_water.agg(lambda v: v.max() - v.min())
    assert spread.idxmax().startswith("7026"), "Andong swings the most"
    andong = d[d.cube.str.startswith("7026")].sort_values("d_water")
    assert (signed(andong.d_water.max()), signed(andong.d_water.min())) == ("+0.703", "+0.247")
    assert andong.head(6).date.str[5:7].isin(WINTER).sum() == 5, \
        "five of its six lowest dates fall between December and March"


def test_gap_filling(runs):
    k = cells(runs)
    qa = pd.read_parquet(POST / "artifacts" / "cell_qa_status.parquet")
    qa["date"] = qa.date.dt.strftime("%Y-%m-%d")
    m = k.merge(qa, left_on=["cube", "date", "fold"], right_on=["cube", "date", "quad"])
    assert len(m) == 384
    synthetic, observed = m[m.fully_synthetic_cell], m[m.fully_observed_cell]
    assert (len(synthetic), f"{len(synthetic) / len(m):.0%}") == (237, "62%"), "62% of the tiles"
    assert signed(synthetic.d_miou.mean()) == "+0.100", "on those filled-in tiles"
    assert (len(observed), signed(observed.d_miou.mean())) == (70, "+0.098"), \
        "on tiles seen entirely on their date"
    assert signed(m.d_miou.corr(m.mean_normalized_gapfill_score)) == "-0.030"


def test_example_frame(runs):
    k = cells(runs)
    frame = k[k.cube.str.startswith("6813") & (k.date == "2018-04-01") & (k.fold == "NW")].iloc[0]
    assert signed(frame.d_miou) == "+0.136"
    assert f"{(k.d_miou < frame.d_miou).mean():.0%}" == "82%", "82% of test cells gain less"
    assert f"{frame.d_miou / k.d_miou.median():.1f}" == "1.3"
    seed0 = runs[(runs.epoch == 120) & (runs.seed == 0) & runs.cube.str.startswith("6813")
                 & (runs.date == "2018-04-01") & (runs.fold == "NW")].iloc[0]
    assert (f"{seed0.rgb0_miou:.3f}", f"{seed0.rgbn_miou:.3f}") == ("0.278", "0.415")
    assert (f"{seed0.rgb0_water:.3f}", f"{seed0.rgbn_water:.3f}") == ("0.129", "0.601")


def test_earlier_runs_sensitivity():
    """The runs whose training crops overlapped the held-out quarter, as reported in METHOD.md."""
    old = pd.read_csv(POST / "runs" / "overlapping" / "cell_level.csv")
    k = cells(old)
    assert signed(by_fold(k, "d_miou").mean()) == "+0.063"
    assert ((k.d_miou > 0).sum(), signed(by_fold(k, "d_water").mean())) == (366, "+0.252")


def test_figure3_shares_record():
    """Figure 3's shares need the checkpoints; figures.py checks its render against this record."""
    import json
    rec = json.loads((POST / "runs" / "figure3_shares.json").read_text())
    assert (rec["only_near_infrared_right_pct"], rec["only_color_right_pct"]) == (14.9, 6.6)
