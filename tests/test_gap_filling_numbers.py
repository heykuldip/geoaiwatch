"""The numbers in the November 2026 post on gap filling and its METHOD.md that come from the
post's results, recomputed from the results files: the tables, audit.json and expectations.json,
with the near-infrared post's cell_qa_status.parquet for the records the two posts share.

"Filling in the days: one pixel at a time". Shares are rounded as the post rounds them: whole
percentages, with the decimals a share needs when it would otherwise read 0% or 100%
(series_figures.pct); a share given as a lower bound is rounded down wherever it appears
(series_figures.pct_at_least). Places run Renshou, Fengcheng, Yangzhou, Andong.
"""
import json
import math
import re
from fractions import Fraction
from pathlib import Path

import pandas as pd
import pytest
from gap_filling_fixtures import scripts_on_path

pytest.importorskip("matplotlib")
scripts_on_path()
import series_figures as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POST = ROOT / "posts" / "2026-11-gap-filling"
RUNS = POST / "runs"
POST1 = ROOT / "posts" / "2026-11-near-infrared" / "artifacts" / "cell_qa_status.parquet"
PLACES = ["Renshou", "Fengcheng", "Yangzhou", "Andong"]
CUBES = ["6466_3380_13", "6730_3430_13", "6813_3313_13", "7026_3201_13"]
TWO_YEARS = "2018-01..2019-12"
TILE = 1024 * 1024
QUARTER = 512 * 512


def read(name):
    return pd.read_csv(RUNS / name)


def by_place(df):
    return df.set_index("place").loc[PLACES]


def shares(num, den):
    return [F.pct(int(n), int(d)) for n, d in zip(num, den)]


def share(x):
    """A share stored as a float, rounded as series_figures.pct rounds counts."""
    return F.pct(*float(x).as_integer_ratio())


@pytest.fixture(scope="module")
def years():
    m = read("qa_monthly.csv")
    return by_place(m[m.period == TWO_YEARS])


@pytest.fixture(scope="module")
def label():
    return read("label_summary.csv").iloc[0]


@pytest.fixture(scope="module")
def frames():
    return read("label_frames.csv")


@pytest.fixture(scope="module")
def audit():
    return json.loads((RUNS / "audit.json").read_text())


@pytest.fixture(scope="module")
def expectations():
    listed = json.loads((RUNS / "expectations.json").read_text())["expectations"]
    return {e["id"]: e for e in listed}


@pytest.fixture(scope="module")
def complete():
    w = read("windows.csv")
    w = w[w.complete].copy()
    w[["n_pixels", "n_covered"]] = w[["n_pixels", "n_covered"]].astype(int)
    return w


def plan():
    return json.loads((POST / "artifacts" / "plan.json").read_text())


def test_two_years_at_four_places(years):
    assert list(years.n_days) == [730] * 4


def test_the_near_infrared_posts_entirely_filled_test_tiles():
    cells = pd.read_parquet(POST1, columns=["fully_synthetic_cell"])
    assert (int(cells.fully_synthetic_cell.sum()), len(cells)) == (237, 384)
    assert F.pct(237, 384) == "62%"


def test_about_a_quarter_fully_actual_and_no_nodata_code(years, expectations):
    assert F.pct(int(years.n_fully_actual.sum()), int(years.n_pixel_days.sum())) == "24%"
    assert shares(years.n_fully_actual, years.n_pixel_days) == ["13%", "21%", "25%", "36%"]
    assert int(years.n_nodata.sum()) == 0
    assert expectations["EX1"]["observed"]["nodata_pixel_days"] == 0
    assert plan()["data"]["nodata"] == plan()["inventory"]["per_file"]["nodata"] == -1000


def test_days_with_every_pixel_and_with_none(years):
    assert list(years.days_all) == [23, 73, 84, 128]
    assert shares(years.days_none, years.n_days) == ["81%", "68%", "63%", "48%"]
    assert list(years.days_none > years.n_days / 2) == [True, True, True, False]


def test_same_day_sentinel2_and_landsat8_codes_on_fully_actual_pixel_days(years, audit):
    diagnostics = audit["diagnostics"]
    codes = [d["l2_codes_on_fully_actual"] for d in diagnostics.values()]
    total = sum(sum(c.values()) for c in codes)
    assert total == int(years.n_fully_actual.sum()) == 733_982_479
    assert F.pct(sum(c.get("900", 0) + c.get("-900", 0) for c in codes), total) == "14%"
    by_code = [F.pct(sum(c.get(k, 0) for c in codes), total) for k in ("0", "900", "-900")]
    assert by_code == ["86%", "11%", "3%"]
    for d in diagnostics.values():
        l2, l3 = d["l2_codes_on_fully_actual"], d["l3_codes_on_fully_actual"]
        assert l3["1"] == l2["0"] and set(l3) <= {"-999", "1", "2", "3", "4", "5", "6"}


def test_the_two_gap_measures():
    g = read("gaps.csv")
    pixel = g[g.level == "pixel"]
    pixel_days = by_place(pixel[pixel.weight == "pixel_days"])
    each_gap = by_place(pixel[pixel.weight == "gaps"])
    assert list(pixel_days["median"]) == [19, 11, 8, 6]
    assert list(each_gap["median"]) == [6, 3, 3, 2]
    assert list(pixel_days["max"]) == [121, 62, 50, 42]
    assert list(each_gap["max"]) == [121, 62, 50, 42]


def test_the_cloud_mask_on_filled_pixel_days(years):
    def both(status):
        return years[f"n_entirely_filled_l3_{status}"] + years[f"n_blended_l3_{status}"]

    assert shares(both("no_planetscope"), years.n_filled) == ["91%", "84%", "82%", "72%"]
    assert shares(both("cloud_shadow_haze"), years.n_filled) == ["7%", "12%", "14%", "21%"]
    assert shares(both("clear"), years.n_filled) == ["3%", "4%", "5%", "7%"]
    assert shares(years.n_blended_l3_clear, years.n_blended) == ["79%", "63%", "70%", "76%"]
    assert [share(x) for x in years.share_dw_blended_l3_clear] == ["81%", "66%", "73%", "78%"]
    assert (years.n_blended_l3_clear * 2 > years.n_blended).all()        # most, by pixels
    assert (years.share_dw_blended_l3_clear > 0.5).all()                 # and by days


def test_the_cloud_mask_weighted_by_date(years):
    """Each day's share of its filled pixels, averaged over the days with any, as the run weights
    entirely filled and blended pixels apart."""
    days = read("qa_daily.csv").drop(columns=["cube", "quarter"]).groupby(["place", "date"]).sum()
    codes = {"no_planetscope": ["no_planetscope"], "clear": ["clear"],
             "cloud_shadow_haze": [f"code{i}" for i in range(2, 7)], "misaligned": ["misaligned"]}
    weighted, counts = {s: [] for s in codes}, []
    for p in PLACES:
        t = days.loc[p]
        t = t[t.n_filled > 0]
        counts.append(len(t))
        for s, cs in codes.items():
            n = sum(t[f"{kind}_l3_{c}"] for kind in ("entirely_filled", "blended") for c in cs)
            each = [Fraction(int(a), int(b)) for a, b in zip(n, t.n_filled)]
            weighted[s].append(sum(each) / len(each))
    assert counts == [730 - n for n in years.days_all] == [707, 657, 646, 602]
    words = {s: [F.pct(m.numerator, m.denominator) for m in v] for s, v in weighted.items()}
    assert words["no_planetscope"] == ["82%", "75%", "71%", "57%"]
    assert words["cloud_shadow_haze"] == ["9%", "14%", "16%", "23%"]
    assert words["clear"] == ["9%", "11%", "13%", "20%"]
    assert max(weighted["misaligned"]) < Fraction(3, 10_000)            # under 0.03%
    misaligned = years.n_entirely_filled_l3_misaligned + years.n_blended_l3_misaligned
    assert (misaligned * 5_000 < years.n_filled).all()                   # under 0.02% by pixels


def test_the_label_date_frames(label, frames):
    assert list(frames.groupby("place").size().loc[PLACES]) == [24] * 4
    assert F.pct(int(label.n_fully_actual), int(label.n_pixels)) == "29%"
    assert F.pct(int(label.n_filled), int(label.n_pixels)) == "71%"
    all_1, all_100 = int(frames.l1_all_1.sum()), int(frames.l1_all_100.sum())
    assert (all_1, all_100, len(frames) - all_1 - all_100) == (14, 55, 27)


def test_the_layer2_sources_of_filled_pixels(label):
    same_day = (label.n_l2_same_day_0 + label.n_l2_same_day_900
                + label.n_l2_same_day_minus900)
    future, pixels, filled = int(label.n_l2_future), int(label.n_pixels), int(label.n_filled)
    assert [F.pct(int(n), filled) for n in (label.n_l2_past, same_day)] == ["45%", "3%"]
    assert F.pct_at_least(future, filled) == "51.6%"                    # later, a bound
    each_frame = [share(label[c]) for c in
                  ("share_dw_l2_past", "share_dw_l2_same_day", "share_dw_l2_future")]
    assert each_frame == ["41%", "11%", "48%"]
    assert int(label.n_frames_with_filled) == 82
    assert F.pct_at_least(future, pixels) == "36.6%"                    # of all the pixels


def test_how_far_the_sources_lie(label):
    assert (label.past_offset_median, label.future_offset_median) == (-3, 3)
    assert (label.past_offset_p90, label.future_offset_p90) == (-7, 11)
    assert (label.past_offset_max, label.future_offset_max) == (-63, 93)


def test_frame_by_frame(label, frames):
    filled = frames[frames.n_filled > 0]
    assert len(filled) == label.n_frames_with_filled == 82
    assert int((filled.n_l2_future > 0).sum()) == 66
    assert int((frames.n_l2_future == frames.n_pixels).sum()) == 13
    assert share(label.median_share_future_of_pixels) == "10%"
    assert int((frames.n_l2_future == 0).sum()) == 30


def test_the_example_frame(label, frames):
    assert (label.example_place, label.example_date) == ("Yangzhou", "2018-01-01")
    assert label.example_n_l2_future == label.example_n_pixels == TILE
    row = frames[(frames.cube == label.example_cube) & (frames.date == label.example_date)]
    assert int(row.n_entirely_filled.iloc[0]) == TILE and int(row.future_offset_p90.iloc[0]) == 13
    tied = frames[frames.n_l2_future == frames.n_pixels]
    assert (len(tied), tied.date.min()) == (13, label.example_date)
    codes = read("label_codes.csv")
    n = codes[(codes.cube == label.example_cube)
              & (codes.date == label.example_date)].groupby("l2").n_filled.sum()
    assert (n.index.min(), n.index.max(), int(n.sum())) == (8, 43, TILE)
    assert int(n[n.index <= 13].sum()) >= 0.9 * TILE                    # nine in ten by 13 days
    assert int(n[(n.index > 13) & (n.index < 43)].sum()) == 0           # the patches at 43
    assert sorted(n.index) == [8, 10, 11, 12, 13, 43]
    assert (int(n[n.index <= 13].sum()), int(n[43])) == (998_692, 49_884)
    assert (int(row.future_offset_median.iloc[0]), int(row.future_offset_p90.iloc[0])) == (10, 13)


def test_fully_actual_days_before_the_label_dates():
    w = read("windows_summary.csv")
    own = w[w.dates == "own"]

    def window(name):
        return by_place(own[own.window == name])

    d15, d30, d60 = window("nested_15"), window("nested_30"), window("nested_60")
    assert shares(d15.n_covered, d15.n_pixels) == ["69%", "88%", "86%", "92%"]
    assert shares(d30.n_covered, d30.n_pixels) == ["94%", "96%", "99%", "99.98%"]
    assert shares(d60.n_covered, d60.n_pixels) == ["99.6%", "100%", "100%", "100%"]
    assert list(d30.days_p50) == [4, 6, 7, 11]
    assert list(d15.n_label_dates) == [23] * 4 and list(d60.n_label_dates) == [21] * 4
    assert list(d15.n_cells) == list(d30.n_cells) == [92] * 4 and list(d60.n_cells) == [84] * 4
    common = w[w.dates == "common"]
    assert set(common.n_label_dates) == {21} and set(common.n_cells) == {84}
    windows = read("windows.csv")
    every = windows.groupby(["cube", "date"]).complete.all()
    assert every.groupby("cube").sum().tolist() == [21] * 4
    dates = every[every].index.get_level_values("date")
    assert (dates.min(), dates.max()) == ("2018-04-01", "2019-12-01")
    complete = windows[windows.complete]
    first = complete.groupby("window").date.min()
    assert first.nested_15 == first.nested_30 == "2018-02-01" and first.nested_60 == "2018-04-01"


def test_many_quarters_at_none_in_15_days_on_the_common_dates(complete):
    dates = [set(complete[complete.window == w].date) for w in complete.window.unique()]
    common = complete[complete.date.isin(set.intersection(*dates))]
    d15 = common[common.window == "nested_15"]
    assert d15.groupby("cube").size().loc[CUBES].tolist() == [84] * 4
    assert [int(((d15.cube == c) & (d15.n_covered == 0)).sum()) for c in CUBES] == [22, 4, 0, 0]


def test_the_eight_files_checked_for_embedded_dates(audit):
    meta = audit["inputs"]["metadata"]
    files = {(k.split()[0], re.sub(r"_", "-", k.split()[1])) for k in meta}
    assert files == {(c, f"{d}.tif") for c in CUBES for d in ("2018-01-01", "2019-12-31")}
    assert all(v["beyond_image_structure"] == [] for v in meta.values())
    assert all(v["descriptions"] == [None] * 9 for v in meta.values())


# the further numbers METHOD.md gives

def test_one_file_a_day_730_days_in_a_row(audit):
    assert audit["inputs"]["pf_qa_files"] == dict.fromkeys(CUBES, 730)
    assert sum(audit["inputs"]["pf_qa_files"].values()) == 2_920
    assert audit["checks"]["I1"] == "passed"
    days = read("qa_daily.csv").groupby("cube").date.agg(["nunique", "min", "max"])
    assert days.loc[CUBES].values.tolist() == [[730, "2018-01-01", "2019-12-31"]] * 4


def test_images_quarters_and_label_dates(frames):
    days = read("qa_daily.csv")
    assert set(days.n_pixels) == {QUARTER} and len(days) == 2_920 * 4
    assert set(frames.n_pixels) == {TILE} and len(frames) == 96
    assert set(frames.date.str[8:]) == {"01"}
    assert frames.groupby("cube").size().loc[CUBES].tolist() == [24] * 4


def test_each_quality_file_has_nine_layers_and_three_are_read(audit):
    assert plan()["inventory"]["per_file"]["bands"] == 9 and audit["checks"]["I2"] == "passed"
    assert list(plan()["data"]["layers_read"]) == ["1", "2", "3"]


def test_where_each_window_is_complete(complete):
    first = complete.groupby("window").date.min()
    dates = complete.groupby(["window", "cube"]).date.nunique().unstack().loc[:, CUBES]
    for name in ("nested_15", "nested_30", "block_1_30"):
        assert first[name] == "2018-02-01" and dates.loc[name].tolist() == [23] * 4
    for name in ("nested_60", "nested_90", "block_31_60", "block_61_90"):
        assert first[name] == "2018-04-01" and dates.loc[name].tolist() == [21] * 4
    assert "2018-01-01" not in set(complete.date) and set(complete.n_pixels) == {QUARTER}


def test_the_expectations_and_how_each_fared(years, expectations):
    assert [expectations[k]["met"] for k in sorted(expectations)] == \
        [True, True, True, False, True, True, True]
    some = expectations["EX4"]["observed"]["days_some"]
    assert [some[c] for c in CUBES] == list(years.days_some) == [117, 160, 183, 253]
    assert F.pct(253, 730) == "35%" and 3 * 253 > 730
    ex2 = expectations["EX2"]["observed"]
    assert F.pct(ex2["fully_actual"], ex2["pixel_days"]) == "24%"
    gaps = expectations["EX5"]["observed"]["median_gap_by_pixel_days"]
    assert [gaps[c] for c in CUBES] == [19, 11, 8, 6]
    ex6 = expectations["EX6"]["observed"]
    assert [F.pct(ex6[c]["no_planetscope"], ex6[c]["filled"]) for c in CUBES] == \
        ["91%", "84%", "82%", "72%"]


def test_the_median_quarter_is_covered_in_30_days(complete, expectations):
    d30 = complete[complete.window == "nested_30"]
    for c in CUBES:
        x = d30[d30.cube == c]
        shares_ = sorted(Fraction(int(a), int(b)) for a, b in zip(x.n_covered, x.n_pixels))
        assert len(shares_) == 92 and shares_[math.ceil(len(shares_) / 2) - 1] == 1
    assert expectations["EX7"]["observed"]["cells"] == dict.fromkeys(CUBES, 92)


def test_days_with_every_pixel_some_and_none(years):
    assert years[["days_all", "days_some", "days_none"]].values.tolist() == \
        [[23, 117, 590], [73, 160, 497], [84, 183, 463], [128, 253, 349]]


def test_the_lowest_and_highest_months():
    m = read("qa_monthly.csv")
    m = m[m.period != TWO_YEARS]
    assert len(m) == 96 and (m.n_fully_actual > 0).all()
    low, high = [], []
    for p in PLACES:
        x = m[m.place == p]
        s = [Fraction(int(a), int(b)) for a, b in zip(x.n_fully_actual, x.n_pixel_days)]
        for out, k in ((low, s.index(min(s))), (high, s.index(max(s)))):
            out.append((x.period.iloc[k], F.pct(s[k].numerator, s[k].denominator)))
    assert low == [("2019-11", "0.8%"), ("2019-02", "2%"), ("2019-02", "8%"), ("2019-07", "9%")]
    assert high == [("2018-04", "32%"), ("2019-09", "61%"), ("2018-10", "47%"),
                    ("2018-02", "65%")]


def test_the_gaps_in_full():
    g = read("gaps.csv")
    pixel = g[g.level == "pixel"]
    pixel_days = by_place(pixel[pixel.weight == "pixel_days"])
    each_gap = by_place(pixel[pixel.weight == "gaps"])
    assert pixel_days[["median", "p90", "max"]].values.tolist() == \
        [[19, 37, 121], [11, 42, 62], [8, 27, 50], [6, 21, 42]]
    assert each_gap[["median", "p90", "max"]].values.tolist() == \
        [[6, 23, 121], [3, 15, 62], [3, 12, 50], [2, 7, 42]]
    assert each_gap[["n_censored_start", "n_censored_end", "n_censored_both"]].values.tolist() == \
        [[1_042_157, 1_048_576, 0], [1_048_576, 1_048_576, 0], [1_048_576, 2_348, 0],
         [77_253, 1_048_576, 0]]


def test_runs_of_days_with_no_fully_actual_pixel():
    g = read("gaps.csv")
    runs = by_place(g[g.level == "tile_none"])
    assert runs[["median", "p90", "max"]].values.tolist() == \
        [[4, 17, 39], [2, 10, 34], [2, 8, 21], [2, 5, 10]]
    assert runs[["n_censored_start", "n_censored_end", "n_censored_both"]].values.tolist() == \
        [[0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0]]


def test_the_label_dates_in_full(label, frames):
    filled = int(label.n_filled)
    codes = ("0", "900", "minus900")
    assert [F.pct(int(label[f"n_l2_same_day_{k}"]), filled) for k in codes] == \
        ["2%", "0.8%", "0.2%"]
    assert [share(label[f"share_dw_l2_same_day_{k}"]) for k in codes] == ["9%", "2%", "0.2%"]
    by_code = read("label_codes.csv")
    far = by_code[by_code.l2.isin([-63, 93])].sort_values("l2")
    assert far[["place", "date", "quarter", "l2", "n_filled"]].values.tolist() == \
        [["Renshou", "2018-07-01", "SW", -63, 7_260], ["Renshou", "2018-08-01", "SW", 93, 5_310]]
    with_filled = frames[frames.n_filled > 0]
    assert int((with_filled.n_l2_past > 0).sum()) == 60
    assert int((2 * with_filled.n_l2_future > with_filled.n_filled).sum()) == 36
    assert int((2 * with_filled.n_l2_past > with_filled.n_filled).sum()) == 37
    assert int((with_filled.n_l2_past == with_filled.n_filled).sum()) == 10


def test_earlier_sources_on_the_first_day_and_after(label):
    by_code = read("label_codes.csv")
    earlier = by_code[(by_code.l2 < 0) & (by_code.l2 > -900)]
    first = earlier[earlier.date == "2018-01-01"]
    assert sorted(set(first.l2)) == [-1] and sorted(set(first.cube)) == \
        [c for c in CUBES if c != label.example_cube]
    assert int(earlier[earlier.date != "2018-01-01"].l2.min()) == -63


def test_the_windows_in_full(complete):
    w = read("windows_summary.csv")
    own = w[w.dates == "own"]

    def window(name):
        return by_place(own[own.window == name])

    d30, d60, d90 = window("nested_30"), window("nested_60"), window("nested_90")
    assert shares(d90.n_covered, d90.n_pixels) == ["99.9%", "100%", "100%", "100%"]
    assert (d30.n_pixels - d30.n_covered).tolist()[3] == 5_070
    assert int(d30.n_pixels.iloc[3]) == 24_117_248
    assert (d60.n_pixels - d60.n_covered).tolist() == [82_185, 0, 0, 0]
    assert (d90.n_pixels - d90.n_covered).tolist() == [14_740, 0, 0, 0]
    d15 = complete[complete.window == "nested_15"]
    assert [int(((d15.cube == c) & (d15.n_covered == 0)).sum()) for c in CUBES] == [22, 5, 4, 0]
    d30_none = complete[(complete.window == "nested_30") & (complete.n_covered == 0)]
    assert d30_none[["place", "date", "quarter"]].values.tolist() == \
        [["Fengcheng", "2019-03-01", "NW"]]
    days = {name: ["/".join(str(int(v)) for v in r)
                   for r in window(name)[["days_p10", "days_p50", "days_p90"]].values]
            for name in ("nested_15", "nested_30", "nested_60", "nested_90")}
    assert days == {"nested_15": ["0/1/5", "0/3/6", "0/4/6", "1/5/10"],
                    "nested_30": ["1/4/8", "2/6/12", "4/7/11", "4/11/17"],
                    "nested_60": ["4/7/13", "5/11/23", "10/14/20", "9/21/31"],
                    "nested_90": ["6/13/18", "8/18/36", "16/22/29", "17/33/45"]}


def test_the_three_blocks():
    b = read("block_coverage.csv")
    t = b.groupby("cube")[["n_pixels", "n_blocks_0", "n_blocks_1", "n_blocks_3"]].sum().loc[CUBES]
    assert shares(t.n_blocks_3, t.n_pixels) == ["90%", "88%", "99%", "99.95%"]
    assert t[["n_blocks_1", "n_blocks_0"]].values.tolist() == \
        [[162_777, 14_740], [0, 0], [0, 0], [0, 0]]
    assert b.groupby("cube").size().loc[CUBES].tolist() == [84] * 4
    short = b[b.n_blocks_3 < b.n_pixels].groupby("cube").size().reindex(CUBES, fill_value=0)
    assert short.tolist() == [34, 24, 5, 4]
    lowest = []
    for c in CUBES:
        x = b[b.cube == c].reset_index(drop=True)
        s = [Fraction(int(n), int(d)) for n, d in zip(x.n_blocks_3, x.n_pixels)]
        k = s.index(min(s))
        lowest.append([x.date[k], x.quarter[k], F.pct(int(x.n_blocks_3[k]), int(x.n_pixels[k]))])
    assert lowest == [["2019-12-01", "SW", "2%"], ["2019-03-01", "NW", "0%"],
                      ["2018-04-01", "SW", "55%"], ["2019-11-01", "SW", "99%"]]


def test_the_cloud_mask_entirely_filled_and_blended_apart(years):
    assert shares(years.n_entirely_filled_l3_no_planetscope, years.n_entirely_filled) == \
        ["93%", "87%", "85%", "77%"]
    assert [share(x) for x in years.share_dw_entirely_filled_l3_no_planetscope] == \
        ["82%", "77%", "73%", "59%"]
    assert shares(years.n_blended, years.n_filled) == ["2%", "4%", "5%", "7%"]


def test_the_label_dates_match_the_near_infrared_posts_records():
    cells = read("label_cells.csv")
    nov = pd.read_parquet(POST1).rename(columns={"quad": "quarter"})
    nov["date"] = pd.to_datetime(nov.date).dt.strftime("%Y-%m-%d")
    both = cells.merge(nov, on=["cube", "date", "quarter"], validate="1:1")
    assert len(both) == len(cells) == 384
    assert ((both.n_fully_actual == QUARTER) == both.fully_observed_cell).all()
    assert ((both.n_entirely_filled == QUARTER) == both.fully_synthetic_cell).all()
    frames = nov.groupby(["cube", "date"])[["fully_observed_cell", "fully_synthetic_cell"]].all()
    assert frames.sum().tolist() == [14, 55] and len(frames) == 96


def test_the_run_record(audit):
    assert len((RUNS / "inputs.sha256").read_text().splitlines()) == 2_921
    assert list(audit["checks"]) == [f"I{i}" for i in range(1, 14)] + \
        ["pf_sr", "blocks_sum", "gap_days", "tables"]
    assert set(audit["checks"].values()) == {"passed"}
    assert round(audit["seconds"]["total"] / 60) == 8
    software = {k: audit["software"][k] for k in ("python", "numpy", "pandas", "rasterio", "gdal")}
    assert software == {"python": "3.11.16", "numpy": "1.26.4", "pandas": "3.0.5",
                        "rasterio": "1.4.4", "gdal": "3.12.3"}

