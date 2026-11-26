"""The gap-filling post's figures, drawn from a synthetic run: what each figure states, checked
against the synthetic series directly; and each of the render's checks, shown to stop it with
nothing written."""
import datetime as dt
import json
import math
import re
import shutil
import subprocess
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from gap_filling_fixtures import (TILE, expected_for, make_series, qa_path, scripts_on_path,
                                  small_design, write_dataset, write_post1_table)

pytest.importorskip("matplotlib")
scripts_on_path()
import series_figures as F  # noqa: E402
import series_plan as SP  # noqa: E402
import series_run as R  # noqa: E402

PLAN = json.loads(SP.PLAN.read_text())
RECORD = {"plan": PLAN, "plan_sha256": SP.PLAN_SHA256, "amendments": [{"id": 1}],
          "amendments_sha256": None}
PNG = b"\x89PNG\r\n\x1a\n"


def quiet(_):
    pass


def captions_for(design, series) -> F.Captions:
    """The figures' caption values, computed from the synthetic series itself, not the tables."""
    shares = {}
    for p in design.places:
        a = series[p.cube]
        nod = (a[:, 0] == design.nodata) | (a[:, 1] == design.nodata) | (a[:, 2] == design.nodata)
        shares[p.cube] = F.pct(int((~nod & (a[:, 0] == 1)).sum()), a[:, 0].size)
    frames, nodata = [], 0
    for p in design.places:
        for day in design.label_dates:
            l1, l2, l3 = series[p.cube][design.index(day)]
            nod = (l1 == design.nodata) | (l2 == design.nodata) | (l3 == design.nodata)
            later = ~nod & (l1 >= 2) & (l2 >= 1) & (l2 <= 899)
            nodata += int(nod.sum())
            frames.append((Fraction(int(later.sum()), l1.size), day.isoformat(),
                           int(p.cube.split("_")[0]), p.cube, l1, l2, later))
    top = max(f[0] for f in frames)
    tied = sorted((f for f in frames if f[0] == top), key=lambda f: (f[1], f[2]))
    share, date, _, cube, l1, l2, later = tied[0]
    median = sorted(f[0] for f in frames)[math.ceil(len(frames) / 2) - 1]   # inverted CDF
    longest = max(w.first for w in design.windows)
    common = [d for d in design.label_dates if d - dt.timedelta(days=longest) >= design.days[0]]
    return F.Captions(
        shares=shares, nodata=nodata, median_later=F.pct(median.numerator, median.denominator),
        example={"cube": cube, "date": date, "tied": len(tied),
                 "offsets": [int(l2[later].min()), int(l2[later].max())],
                 "entirely_filled": bool((l1 == 100).all())},
        common={"dates": len(common), "first": common[0].isoformat(),
                "cells": len(common) * len(design.quarters)})


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    design = small_design()
    series = make_series(design, seed=0)
    root = tmp_path_factory.mktemp("gap_filling_figures")
    den = root / "den"
    write_dataset(den, design, series)
    rng = np.random.default_rng(1)
    for path in sorted((den / "planet").rglob("PF-SR/*.tif")):     # a picture worth stretching
        with rasterio.open(path, "r+") as f:
            f.write(rng.integers(200, 3000, (4, TILE, TILE)).astype("uint16"))
    write_post1_table(root / "post1.parquet", design, series)
    R.run(design, RECORD, den, root / "runs", root / "post1.parquet",
          expected_for(design, series), log=quiet)
    captions = captions_for(design, series)
    record = F.render(design, root / "runs", den, root / "figures", captions, log=quiet)
    return SimpleNamespace(design=design, series=series, root=root, den=den, captions=captions,
                           record=record)


def copy_runs(syn, tmp_path):
    runs = tmp_path / "runs"
    shutil.copytree(syn.root / "runs", runs)
    return runs


def edit(path, old, new):
    text = path.read_text()
    assert text.count(old) >= 1
    path.write_text(text.replace(old, new, 1))


def stops(syn, tmp_path, runs=None, captions=None, check=None, **kw):
    out = tmp_path / "figures"
    with pytest.raises(F.Q.CheckFailed) as e:
        F.render(syn.design, runs or syn.root / "runs", syn.den, out, captions or syn.captions,
                 log=quiet, **kw)
    assert not out.exists() or not any(out.iterdir())
    assert not list(tmp_path.glob(".figures-*"))
    if check:
        assert e.value.check == check
    return e.value


def test_every_figure_is_written(syn):
    out = syn.root / "figures"
    assert {p.name for p in out.iterdir()} == set(F.OUTPUTS)
    for name in F.FIGS:
        data = (out / name).read_bytes()
        assert data[:8] == PNG
        assert syn.record["outputs"][name] == F.sha256(data)
    rec = json.loads((out / F.RECORD).read_text())
    assert rec["checks"].pop("commit") == "not checked (git not used)"
    assert set(rec["checks"].values()) == {"passed"}
    assert list(rec["captions"]) == ["shares", "nodata", "median_later", "example", "common"]
    assert rec["captions"]["example"]["cube"] == syn.captions.example["cube"]
    assert set(rec["runs"]["files"]) == set(F.READS) and "window_days.csv" in F.READS
    assert len(rec["dataset"]) == 2
    assert rec["code"]["git"] == "not used" and "matplotlib" in rec["software"]
    assert not list(syn.root.glob(".figures-*"))


def test_the_planted_tie_is_the_example_frame(syn):
    assert syn.captions.example["tied"] >= 3
    assert (syn.captions.example["cube"], syn.captions.example["date"]) == \
        ("6466_3380_13", "2018-05-01")
    assert syn.captions.example["offsets"] == [5, 5]


@pytest.mark.parametrize("num, den, words", [
    (0, 7, "0%"), (7, 7, "100%"), (1, 3, "33%"), (1, 8, "13%"), (1, 400, "0.3%"),
    (81, 10_000, "0.8%"), (4, 10_000, "0.04%"), (996_268, 1_000_000, "99.6%"),
    (24_112_178, 24_117_248, "99.98%"), (1_048_575, 1_048_576, "99.9999%"),
    (133_004, 1_000_000, "13%"), (98_223, 1_000_000, "10%"), (995, 1000, "99.5%")])
def test_shares_are_never_rounded_to_all_or_none(num, den, words):
    assert F.pct(num, den) == words


@pytest.mark.parametrize("num, den", [(1, 0), (-1, 3), (4, 3)])
def test_a_share_outside_zero_to_one_is_refused(num, den):
    with pytest.raises(ValueError):
        F.pct(num, den)
    with pytest.raises(ValueError):
        F.pct_at_least(num, den)


@pytest.mark.parametrize("num, den, decimals, words", [
    (366_258, 1_000_000, 1, "36.6%"), (2, 3, 1, "66.6%"), (2, 3, 0, "66%"), (1, 2, 1, "50.0%"),
    (1, 3, 2, "33.33%"), (0, 7, 1, "0.0%"), (7, 7, 1, "100.0%")])
def test_a_lower_bound_is_rounded_down(num, den, decimals, words):
    assert F.pct_at_least(num, den, decimals) == words
    assert Fraction(words.rstrip("%")) <= Fraction(100 * num, den)


@pytest.mark.parametrize("k, words", [
    (0, ""),
    (1, "1 pixel carries the nodata code in quality layers 1–3."),
    (1234, "1,234 pixels carry the nodata code in quality layers 1–3.")])
def test_nodata_pixels_are_named_by_the_code(k, words):
    assert F.nodata_words(k) == words


def test_the_layer2_map_colors_each_category():
    from matplotlib import colormaps
    from matplotlib.colors import to_rgba
    l1 = np.array([[1, 100, 100, 100], [100, 100, 50, -1000]], np.int16)
    l2 = np.array([[0, 0, 900, -900], [-10, 20, 3, 5]], np.int16)
    l3 = np.ones_like(l1)
    c = F.Q.classify(l1, l2, l3, -1000)
    rgba, norm = F.layer2_map(l2, c, F.Q.sources(l2, c.filled))
    assert (norm.vmin, norm.vmax) == (-20, 20)
    assert tuple(rgba[0, 0]) == to_rgba(F.FULLY_ACTUAL)
    assert tuple(rgba[0, 1]) == to_rgba(F.SAME_DAY["same_day_0"])
    assert tuple(rgba[0, 2]) == to_rgba(F.SAME_DAY["same_day_900"])
    assert tuple(rgba[0, 3]) == to_rgba(F.SAME_DAY["same_day_minus900"])
    cmap = colormaps[F.OFFSETS]
    assert np.allclose(rgba[1, 0], cmap(norm(-10.0)))
    assert np.allclose(rgba[1, 1], cmap(norm(20.0)))
    assert rgba[1, 0][2] > rgba[1, 0][0] and rgba[1, 1][0] > rgba[1, 1][2]   # purple, orange
    assert tuple(rgba[1, 3]) == to_rgba(F.NODATA)


def test_true_color_stretches_each_band():
    rgb = np.stack([np.arange(100).reshape(10, 10) * k for k in (10, 20, 30)]).astype(np.int16)
    img = F.true_color(rgb)
    assert img.shape == (10, 10, 3) and img.min() == 0 and img.max() == 1
    assert np.allclose(img[..., 0], img[..., 2])                        # each band on its own


def test_the_example_frames_caption_discloses_the_tie():
    design = small_design()
    f = {"date": "2018-01-01", "n_frames": 96,
         "printed": {"offsets": [8, 43], "entirely_filled": True, "share": "100%", "tied": 13,
                     "median": "10%", "first_day": True}}
    text = F.frame_caption(design, "Yangzhou", f)
    assert "Every pixel is entirely gap-filled, and Layer 2 records a later source for every " \
           "pixel, 8 to 43 days later." in text
    assert "13 of the 96 images are at 100%" in text and "the first day of the series." in text
    assert "the median share is 10%." in text and F.ATTRIB in text
    assert "· the whole image, " in text
    assert "the image with the largest share of its pixels filled and with a later Layer 2 " \
           "source." in text
    f["printed"] |= {"entirely_filled": False, "share": "37%", "tied": 1, "first_day": False}
    text = F.frame_caption(design, "Andong", f)
    assert "For 37% of its pixels, Layer 2 records a later source, 8 to 43 days later." in text
    assert "images are at" not in text and "first day" not in text
    assert "future" not in text.lower()                                 # earlier and later


def test_a_changed_count_stops_the_render(syn, tmp_path):
    runs = copy_runs(syn, tmp_path)
    p = runs / "label_frames.csv"
    rows = p.read_text().splitlines()
    head = rows[0].split(",")
    i = head.index("n_l2_past")
    cells = rows[5].split(",")
    cells[i] = str(int(cells[i]) + 1)
    rows[5] = ",".join(cells)
    p.write_text("\n".join(rows) + "\n")
    stops(syn, tmp_path, runs, check="figure 2")


def test_a_wrong_pf_qa_hash_stops_the_render(syn, tmp_path):
    runs = copy_runs(syn, tmp_path)
    cube, date = syn.captions.example["cube"], syn.captions.example["date"]
    rel = qa_path(syn.den, cube, dt.date.fromisoformat(date)).relative_to(syn.den).as_posix()
    p = runs / "inputs.sha256"
    line = next(x for x in p.read_text().splitlines() if x.endswith(rel))
    edit(p, line, "0" * 64 + "  " + rel)
    stops(syn, tmp_path, runs, check="figure 3")


def test_a_changed_pf_sr_file_stops_the_render(syn, tmp_path):
    den = tmp_path / "den"
    shutil.copytree(syn.den, den)
    day = dt.date.fromisoformat(syn.captions.example["date"])
    sr = qa_path(den, syn.captions.example["cube"], day).parent.parent / "PF-SR" / f"{day}.tif"
    with rasterio.open(sr, "r+") as f:
        f.write(np.zeros((4, TILE, TILE), "uint16"))
    out = tmp_path / "figures"
    with pytest.raises(F.Q.CheckFailed) as e:
        F.render(syn.design, syn.root / "runs", den, out, syn.captions, log=quiet)
    assert e.value.check == "figure 3" and not out.exists()


def test_a_missing_cell_stops_the_render(syn, tmp_path):
    runs = copy_runs(syn, tmp_path)
    p = runs / "windows.csv"
    rows = p.read_text().splitlines()
    k = next(i for i, x in enumerate(rows) if ",2019-06-01,NW,nested_60," in x)
    p.write_text("\n".join(rows[:k] + rows[k + 1:]) + "\n")
    stops(syn, tmp_path, runs, check="figure 4")


def test_a_daily_count_that_misses_the_two_years_stops_the_render(syn, tmp_path):
    runs = copy_runs(syn, tmp_path)
    p = runs / "qa_monthly.csv"
    rows = p.read_text().splitlines()
    head = rows[0].split(",")
    i = head.index("n_fully_actual")
    k = next(j for j, x in enumerate(rows) if ",2018-01..2019-12," in x)
    cells = rows[k].split(",")
    cells[i] = str(int(cells[i]) - 1)
    rows[k] = ",".join(cells)
    p.write_text("\n".join(rows) + "\n")
    stops(syn, tmp_path, runs, check="figure 1")


def set_cell(path, match, column, value):
    rows = path.read_text().splitlines()
    i = rows[0].split(",").index(column)
    k = next(j for j, x in enumerate(rows) if match(x.split(",")))
    cells = rows[k].split(",")
    cells[i] = value(cells[i])
    rows[k] = ",".join(cells)
    path.write_text("\n".join(rows) + "\n")


def test_nodata_that_differs_from_the_daily_table_stops_the_render(syn, tmp_path):
    runs = copy_runs(syn, tmp_path)
    set_cell(runs / "qa_daily.csv", lambda c: c[2] == "2019-02-01", "n_nodata",
             lambda v: str(int(v) + 1))
    e = stops(syn, tmp_path, runs, check="figure 2")
    assert "qa_daily.csv" in str(e)


def test_label_codes_that_move_the_largest_share_stop_the_render(syn, tmp_path):
    cube, date = "6730_3430_13", "2018-05-01"                 # tied, not the example frame
    assert (cube, date) != (syn.captions.example["cube"], syn.captions.example["date"])
    runs = copy_runs(syn, tmp_path)
    set_cell(runs / "label_codes.csv", lambda c: (c[0], c[2]) == (cube, date), "l2",
             lambda v: str(-int(v)))
    e = stops(syn, tmp_path, runs, check="figure 3")
    assert "label_codes.csv" in str(e)


LEFT_OUT = {"a quarter": lambda x: ",2019-06-01,NW,nested_30," in x,
            "a label date in one window": lambda x: ",2018-04-01," in x and ",block_61_90," in x}


def test_a_covered_pixel_counted_in_window_days_as_uncovered_stops_the_render(syn, tmp_path):
    runs = copy_runs(syn, tmp_path)
    set_cell(runs / "window_days.csv",
             lambda c: c[2:5] == ["2019-06-01", "NW", "nested_30"] and c[5] != "0",
             "fully_actual_days", lambda v: "0")
    e = stops(syn, tmp_path, runs, check="figure 4")
    assert "window_days.csv" in str(e)


@pytest.mark.parametrize("left_out", sorted(LEFT_OUT))
def test_rows_left_out_of_window_days_stop_the_render(syn, tmp_path, left_out):
    runs = copy_runs(syn, tmp_path)
    p = runs / "window_days.csv"
    rows = p.read_text().splitlines()
    keep = [x for x in rows if not LEFT_OUT[left_out](x)]
    assert len(keep) < len(rows)
    p.write_text("\n".join(keep) + "\n")
    e = stops(syn, tmp_path, runs, check="figure 4")
    assert "window_days.csv" in str(e)


@pytest.mark.parametrize("field, value, check", [
    ("shares", {"6466_3380_13": "1%"}, "figure 1"),
    ("nodata", "one more", "figure 2"),
    ("median_later", "99%", "figure 3"),
    ("example", {"tied": 2}, "figure 3"),
    ("example", {"offsets": [4, 5]}, "figure 3"),
    ("example", {"cube": "7026_3201_13"}, "figure 3"),
    ("common", {"dates": 20}, "figure 4"),
    ("common", {"cells": 83}, "figure 4")])
def test_a_caption_value_off_the_tables_stops_the_render(syn, tmp_path, field, value, check):
    old = getattr(syn.captions, field)
    new = {**old, **value} if isinstance(old, dict) else old + 1 if value == "one more" else value
    stops(syn, tmp_path, captions=F.Captions(**{**F.asdict(syn.captions), field: new}),
          check=check)


def test_figures_already_there_are_refused_unless_replaced(syn, tmp_path):
    out = tmp_path / "figures"
    out.mkdir()
    (out / F.FIGS[0]).write_bytes(b"old")
    with pytest.raises(F.Q.CheckFailed) as e:
        F.render(syn.design, syn.root / "runs", syn.den, out, syn.captions, log=quiet)
    assert e.value.check == "outputs" and (out / F.FIGS[0]).read_bytes() == b"old"
    F.render(syn.design, syn.root / "runs", syn.den, out, syn.captions, replace=True, log=quiet)
    assert (out / F.FIGS[0]).read_bytes() == (syn.root / "figures" / F.FIGS[0]).read_bytes()


@pytest.mark.parametrize("state", [{"commit": "a", "clean": False},
                                   {"commit": None, "clean": None}])
def test_a_checkout_that_isnt_clean_is_refused(syn, tmp_path, state):
    stops(syn, tmp_path, git=lambda: state, check="commit")


def test_tables_that_arent_committed_are_refused(tmp_path):
    repo = tmp_path / "repo"
    runs = repo / "runs"
    runs.mkdir(parents=True)
    for name in F.READS:
        (runs / name).write_text("x\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    with pytest.raises(F.Q.CheckFailed) as e:
        F.committed_runs(runs, repo)
    assert e.value.check == "commit"
    subprocess.run(["git", "-C", str(repo), "add", "runs"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "-m", "runs"], check=True)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert F.committed_runs(runs, repo) == head


def test_the_figures_say_image_and_the_nodata_code(syn, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    drawn, close = [], plt.close
    monkeypatch.setattr(plt, "close", drawn.append)
    F.render(syn.design, syn.root / "runs", syn.den, tmp_path / "figures", syn.captions, log=quiet)
    monkeypatch.undo()
    text = " ".join(t.get_text() for fig in drawn for t in fig.findobj(Text))
    legends = [t.get_text() for fig in drawn for leg in fig.legends for t in leg.get_texts()]
    for fig in drawn:
        close(fig)
    assert len(drawn) == 4
    assert not re.search(r"\btiles?\b|missing", text)
    assert "share of the image's pixels fully actual" in text and "An image is " in text
    assert syn.captions.nodata > 0 and F.nodata_words(syn.captions.nodata) in text
    assert "nodata" in legends
    assert "Each dot is a quarter of an image" in text
    assert "the orange line pools the pixels" in text
    assert "map date" in text and "label date" not in text
    assert "has a fully-actual day, not necessarily the same one." in text


def test_the_figures_are_drawn_the_same_each_time(syn, tmp_path):
    out = tmp_path / "again"
    record = F.render(syn.design, syn.root / "runs", syn.den, out, syn.captions, log=quiet)
    assert record["outputs"] == syn.record["outputs"]


def test_figure_3s_legend_names_only_what_the_image_holds():
    none = np.zeros(3, bool)
    some = np.array([False, True, False])
    src = {k: none for k in F.SAME_DAY}
    assert F.frame_handles(SimpleNamespace(fully_actual=none), src) == []
    src = {**src, "same_day_900": some}
    got = [h.get_label() for h in F.frame_handles(SimpleNamespace(fully_actual=some), src)]
    assert got == ["fully actual", F.SAME_DAY_WORDS["same_day_900"]]
