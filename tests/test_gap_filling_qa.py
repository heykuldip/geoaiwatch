"""The gap-filling post's measurements, piece by piece: the definitions at every boundary code,
quantiles, gaps, windows, and the checks on the inputs, each shown to fail on a broken input."""
import dataclasses
import datetime as dt
import hashlib
import itertools

import numpy as np
import pytest
import rasterio
from affine import Affine
from gap_filling_fixtures import TILE, ZONES, qa_path, scripts_on_path, small_design, write_qa

scripts_on_path()
import series_plan as SP  # noqa: E402
import series_qa as Q  # noqa: E402

NODATA = -1000


# definitions

def literal(l1, l2, l3):
    """The plan's tests on one pixel-day, read literally."""
    if NODATA in (l1, l2, l3):
        return "nodata"
    if l1 == 1:
        return "fully_actual"
    if 2 <= l1 <= 100:
        return "entirely_filled" if l1 == 100 else "blended"
    raise ValueError(l1)


def test_classes_at_every_boundary_code():
    combos = list(itertools.product((NODATA, 1, 2, 99, 100),
                                    (NODATA, -900, -899, -1, 0, 1, 899, 900),
                                    (NODATA, -999, 1, 2, 6, 7)))
    l1, l2, l3 = (np.array(x, np.int16) for x in zip(*combos))
    c = Q.classify(l1, l2, l3, NODATA)
    for i, combo in enumerate(combos):
        want = literal(*combo)
        got = [k for k in ("nodata", "fully_actual", "entirely_filled", "blended")
               if getattr(c, k)[i]]
        assert got == [want], combo
        assert c.filled[i] == (want in ("entirely_filled", "blended")), combo
    src = Q.sources(l2, c.filled)
    for i, (_, b, _) in enumerate(combos):
        got = [s for s in Q.SOURCES if src[s][i]]
        if not c.filled[i]:
            assert got == []
            continue
        want = {0: "same_day_0", 900: "same_day_900", -900: "same_day_minus900"}.get(b)
        want = want or ("past" if b < 0 else "future")
        assert got == [want], combos[i]


@pytest.mark.parametrize("layer, code", [(0, 0), (0, 101), (1, 901), (1, -901), (2, 0), (2, 8),
                                         (2, -998)])
def test_codes_out_of_range_are_found(layer, code):
    a = [np.array([1, 100], np.int16), np.array([0, 5], np.int16), np.array([1, -999], np.int16)]
    assert Q.code_problems(*a, NODATA) == []
    a[layer][1] = code
    bad = Q.code_problems(*a, NODATA)
    assert len(bad) == 1 and bad[0].startswith(f"L{layer + 1} ") and str(code) in bad[0]


def test_nodata_is_in_range_in_every_layer():
    a = [np.array([NODATA, 1], np.int16), np.array([NODATA, 0], np.int16),
         np.array([NODATA, 1], np.int16)]
    assert Q.code_problems(*a, NODATA) == []


# quantiles

def test_quantiles_match_numpys_inverted_cdf():
    rng = np.random.default_rng(0)
    for n in list(range(1, 25)) + [92, 96, 100]:
        v = rng.integers(-20, 20, n)
        for k in range(1, 10):
            want = int(np.quantile(v, k / 10, method="inverted_cdf"))
            assert Q.quantile(v, np.ones(n, int), k) == want, (n, k)


def test_weighted_quantiles_are_those_of_the_expanded_sample():
    rng = np.random.default_rng(1)
    for _ in range(200):
        v = rng.integers(0, 30, rng.integers(1, 12))
        w = rng.integers(0, 6, v.size)
        if w.sum() == 0:
            assert Q.quantile(v, w, 5) is None
            continue
        expanded = np.repeat(v, w)
        for k in (1, 5, 9):
            assert Q.quantile(v, w, k) == int(np.quantile(expanded, k / 10, method="inverted_cdf"))
        assert Q.largest(v, w) == int(expanded.max())


def test_a_quantile_at_an_exact_fraction_takes_the_smaller_value():
    assert Q.quantile([1, 2, 3, 4], [1, 1, 1, 1], 5) == 2
    assert Q.quantile([1, 2], [9, 1], 9) == 1


def test_offsets_are_distances_reported_signed():
    h = np.zeros(Q.L2_BINS, np.int64)
    for code, n in ((-1, 5), (-10, 4), (-300, 1), (2, 1), (7, 8), (0, 50), (900, 9), (-900, 9)):
        h[code + Q.L2_OFFSET] = n
    assert Q.offsets(h) == {"past_offset_median": -1, "past_offset_p90": -10,
                            "past_offset_max": -300, "future_offset_median": 7,
                            "future_offset_p90": 7, "future_offset_max": 7}


# gaps

def series_gaps(flags):
    g = Q.Gaps((len(flags[0]),), len(flags))
    for day in flags:
        g.add(np.array(day, bool))
    return g


def test_gaps_and_their_censoring():
    f, t = False, True
    days = [(f, f, t), (f, f, t), (t, f, t), (f, f, t), (t, f, t), (t, f, t), (f, f, t), (f, f, t)]
    g = series_gaps(days)
    g.finish()
    g.check("test")
    nonzero = {k: {i: int(n) for i, n in enumerate(h) if n} for k, h in g.hist.items()}
    assert nonzero == {"none": {1: 1}, "start": {2: 1}, "end": {2: 1}, "both": {8: 1}}
    assert g.fully_actual_days.tolist() == [3, 0, 8]
    assert g.gap_days.tolist() == [5, 8, 0]


def test_a_run_lost_at_the_end_of_the_series_fails_i7():
    g = series_gaps([(False,), (True,), (False,)])
    g.day = g.n_days
    with pytest.raises(Q.CheckFailed, match="I7"):
        g.check("test")


def test_runs_of_class_none():
    h = Q.runs_of([True, False, True, True, False, True], 6)
    assert {k: {i: int(n) for i, n in enumerate(v) if n} for k, v in h.items()} == {
        "none": {2: 1}, "start": {1: 1}, "end": {1: 1}, "both": {}}
    assert {k: v.sum() for k, v in Q.runs_of([True] * 4, 4).items()} == {
        "none": 0, "start": 0, "end": 0, "both": 1}


# windows

def ring_with(design, flags):
    ring = Q.Ring(90, flags.shape[1:])
    for t, fa in enumerate(flags):
        ring.push(t, fa)
    return ring


def test_window_cells_against_a_direct_count():
    design = small_design()
    rng = np.random.default_rng(2)
    flags = rng.random((200, TILE, TILE)) < 0.3
    day = dt.date(2018, 6, 1)
    t = design.index(day)
    qs = Q.quarter_slices(design)
    rows, hist_rows, blocks = Q.window_cells(ring_with(design, flags[:t]), t, day, design, qs)
    assert len(rows) == 28 and all(r["complete"] for r in rows) and len(blocks) == 4
    for r in rows:
        w = next(x for x in design.windows if x.name == r["window"])
        sub = flags[t - w.first:t - w.last + 1][(slice(None),) + qs[r["quarter"]]]
        count = sub.sum(axis=0).ravel()
        assert r["n_covered"] == int((count > 0).sum())
        assert r["n_days_any"] == int(sub.reshape(len(sub), -1).any(axis=1).sum())
        for k, key in ((1, "days_p10"), (5, "days_p50"), (9, "days_p90")):
            assert r[key] == int(np.quantile(count, k / 10, method="inverted_cdf"))
        assert (r["first_day"], r["last_day"]) == (day - dt.timedelta(w.first),
                                                   day - dt.timedelta(w.last))
    for b in blocks:
        sl = (slice(None),) + qs[b["quarter"]]
        k = sum((flags[t - hi:t - lo + 1][sl].sum(axis=0) > 0).astype(int)
                for lo, hi in ((1, 30), (31, 60), (61, 90)))
        want = np.bincount(k.ravel(), minlength=4).tolist()
        assert [b[f"n_blocks_{i}"] for i in range(4)] == want


def test_incomplete_windows_are_listed_without_values():
    design = small_design()
    flags = np.zeros((40, TILE, TILE), bool)
    day = dt.date(2018, 2, 1)
    rows, _, blocks = Q.window_cells(ring_with(design, flags[:31]), 31, day, design,
                                     Q.quarter_slices(design))
    complete = {r["window"] for r in rows if r["complete"]}
    assert complete == {"nested_15", "nested_30", "block_1_30"} and blocks == []
    assert all("n_covered" not in r for r in rows if not r["complete"])


def test_block_1_30_one_day_off_fails_i12(monkeypatch):
    design = small_design()
    flags = np.random.default_rng(3).random((120, TILE, TILE)) < 0.3
    real = Q.Ring.window_days

    def shifted(self, t, w):
        days = real(self, t, w)
        return [d - 1 for d in days] if w.name == "block_1_30" else days

    monkeypatch.setattr(Q.Ring, "window_days", shifted)
    with pytest.raises(Q.CheckFailed, match="I12"):
        Q.window_cells(ring_with(design, flags[:100]), 100, design.days[100], design,
                       Q.quarter_slices(design))


def test_a_blocks_flags_changed_fail_i13(monkeypatch):
    design = small_design()
    flags = np.random.default_rng(4).random((120, TILE, TILE)) < 0.02
    flags[:, 0, 0] = False
    real = Q.Ring.counts

    def altered(self, t, w):
        out = real(self, t, w)
        if w.name == "block_31_60":
            out = out.copy()
            out[0, 0] = 0 if out[0, 0] else 1
        return out

    monkeypatch.setattr(Q.Ring, "counts", altered)
    with pytest.raises(Q.CheckFailed, match="I13"):
        Q.window_cells(ring_with(design, flags[:100]), 100, design.days[100], design,
                       Q.quarter_slices(design))


def test_a_window_reaching_the_label_date_fails_i8():
    design = small_design()
    bad = dataclasses.replace(design, windows=design.windows + (SP.Window("nested_0", 15, 0),))
    flags = np.zeros((40, TILE, TILE), bool)
    with pytest.raises(Q.CheckFailed, match="I8"):
        Q.window_cells(ring_with(design, flags[:31]), 31, dt.date(2018, 2, 1), bad,
                       Q.quarter_slices(design))


# one day's counts

def test_counts_made_inconsistent_fail_i4_and_i5():
    design = small_design()
    rng = np.random.default_rng(5)
    l1 = rng.choice(np.array([1, 2, 50, 100], np.int16), (TILE, TILE))
    l2 = rng.integers(-5, 6, (TILE, TILE)).astype(np.int16)
    l3 = rng.choice(np.array([-999, 1, 2, 7], np.int16), (TILE, TILE))
    l1[0, 0], l3[0, 0] = 50, 1
    c = Q.classify(l1, l2, l3, NODATA)
    counts = Q.day_counts(c, l3, Q.quarter_slices(design))
    Q.check_day(counts, c, design, "test")
    broken = {k: dict(v) for k, v in counts.items()}
    broken["NW"]["n_blended"] += 1
    with pytest.raises(Q.CheckFailed, match="I4"):
        Q.check_day(broken, c, design, "test")
    moved = {k: dict(v) for k, v in counts.items()}
    for k, d in (("n_fully_actual", 1), ("n_filled", -1), ("n_blended", -1),
                 ("blended_l3_clear", -1)):
        moved["NW"][k] += d
    with pytest.raises(Q.CheckFailed, match="I5"):
        Q.check_day(moved, c, design, "test")


# inventory and headers, on a ten-day series

def short_design():
    d = small_design()
    return dataclasses.replace(d, n_days=10, label_dates=(d.first_day,))


def short_dataset(den, design, **change):
    a = np.ones((3, TILE, TILE), np.int16)
    for p in design.places:
        for day in design.days:
            write_qa(qa_path(den, p.cube, day), a, ZONES[p.cube][1])
        sr = qa_path(den, p.cube, design.first_day).parent.parent / "PF-SR"
        sr.mkdir(parents=True, exist_ok=True)
        write_qa(sr / f"{design.first_day}.tif", a, ZONES[p.cube][1])
    return den


def test_a_complete_short_dataset_passes(tmp_path):
    design = short_design()
    den = short_dataset(tmp_path, design)
    (qa_path(den, "6466_3380_13", design.first_day).parent / "notes.aux.xml").write_text("x")
    inv = Q.inventory(design, den / "planet")
    assert inv.others["6466_3380_13"] == ["notes.aux.xml"]
    refs, meta = Q.headers(design, inv)
    assert len(refs) == 4 and len(meta) == 8
    assert all(m["beyond_image_structure"] == [] for m in meta.values())


@pytest.mark.parametrize("break_it, check", [
    (lambda den, d: qa_path(den, "6730_3430_13", d.days[3]).unlink(), "I1"),
    (lambda den, d: qa_path(den, "6466_3380_13", d.days[2]).rename(
        qa_path(den, "6466_3380_13", d.days[2]).with_name(f"{d.days[2]:%Y_%m_%d}.tif")), None),
    (lambda den, d: write_qa(qa_path(den, "6466_3380_13", d.days[2]).with_name(
        f"{d.days[2]:%Y_%m_%d}.tif"), np.ones((3, TILE, TILE)), 32648), "I1"),
    (lambda den, d: write_qa(qa_path(den, "6466_3380_13", d.days[2]).with_name("extra.tif"),
                             np.ones((3, TILE, TILE)), 32648), "I1"),
    (lambda den, d: write_qa(qa_path(den, "6466_3380_13", d.days[2]).with_name("2020-01-01.tif"),
                             np.ones((3, TILE, TILE)), 32648), "I1"),
    (lambda den, d: (qa_path(den, "7026_3201_13", d.first_day).parent.parent / "PF-SR"
                     / f"{d.first_day}.tif").unlink(), None),
])
def test_the_inventory(tmp_path, break_it, check):
    design = short_design()
    den = short_dataset(tmp_path, design)
    break_it(den, design)
    if check is None:
        Q.inventory(design, den / "planet")
        return
    with pytest.raises(Q.CheckFailed, match=check):
        Q.inventory(design, den / "planet")


@pytest.mark.parametrize("change, check", [
    ({"count": 8}, "I2"), ({"dtype": "float32"}, "I2"), ({"width": TILE - 1}, "I2"),
    ({"nodata": None}, "I2"), ({"crs": "EPSG:4326"}, "I2"),
    ({"transform": Affine(3.0, 0.0, 5e5, 0.0, -4.0, 3.3e6)}, "I2"), ("garbage", "I1")])
def test_the_headers(tmp_path, change, check):
    design = short_design()
    den = short_dataset(tmp_path, design)
    path = qa_path(den, "6813_3313_13", design.days[4])
    if change == "garbage":
        path.write_bytes(b"not a GeoTIFF")
    else:
        write_qa(path, np.ones((3, TILE, TILE), np.int16), ZONES["6813_3313_13"][1], **change)
    with pytest.raises(Q.CheckFailed, match=check):
        Q.headers(design, Q.inventory(design, den / "planet"))


def test_metadata_beyond_image_structure_is_reported(tmp_path):
    design = short_design()
    den = short_dataset(tmp_path, design)
    with rasterio.open(qa_path(den, "6466_3380_13", design.days[-1]), "r+") as f:
        f.update_tags(GAPFILL_DATES="20180102,20180105")
    _, meta = Q.headers(design, Q.inventory(design, den / "planet"))
    flagged = {k: v["beyond_image_structure"] for k, v in meta.items()
               if v["beyond_image_structure"]}
    assert flagged == {f"6466_3380_13 {design.days[-1]}.tif": ["default:GAPFILL_DATES"]}


def test_a_changed_file_fails_i2_when_read(tmp_path):
    design = short_design()
    den = short_dataset(tmp_path, design)
    path = qa_path(den, "6466_3380_13", design.days[0])
    with rasterio.open(path) as s:
        ref = (s.crs, s.transform)
    digest, l1, l2, l3 = Q.read_day(path, design, ref)
    assert len(digest) == 64 and l1.shape == (TILE, TILE)
    write_qa(path, np.ones((3, TILE, TILE), np.int16), 32650)
    with pytest.raises(Q.CheckFailed, match="I2"):
        Q.read_day(path, design, ref)


# the example frame's PF-SR file

def write_sr(path, count=4, size=TILE, transform=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=size, height=size, count=count,
                       dtype="int16", crs="EPSG:32648", compress="lzw",
                       transform=transform or Affine(3.0, 0.0, 5e5, 0.0, -3.0, 3.3e6)) as f:
        f.write(np.random.default_rng(0).integers(1, 10_000, (count, size, size), "int16"))


def truncated(path):
    """A PF-SR file whose header opens but whose pixels don't decode."""
    write_sr(path)
    path.write_bytes(path.read_bytes()[:int(path.stat().st_size * 0.7)])


def pf_sr_case(tmp_path):
    design = short_design()
    den = short_dataset(tmp_path, design)
    folder = qa_path(den, "6466_3380_13", design.first_day).parent.parent / "PF-SR"
    for f in folder.iterdir():
        f.unlink()
    with rasterio.open(qa_path(den, "6466_3380_13", design.first_day)) as s:
        ref = (s.crs, s.transform)
    return design, den, folder, ref


def test_the_example_frames_pf_sr_is_found_read_and_hashed(tmp_path):
    design, den, folder, ref = pf_sr_case(tmp_path)
    write_sr(folder / f"{design.days[3]:%Y_%m_%d}.tif")
    write_sr(folder / f"{design.days[4]}.tif", transform=Affine(3.0, 0.0, 6e5, 0.0, -3.0, 3.3e6))
    path = Q.find_pf_sr(den / "planet", "6466_3380_13", design.days[3])
    info = Q.read_pf_sr(path, design, ref)
    assert info["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (info["bands"], info["pixels"], info["same_grid_as_pf_qa"]) == (4, [TILE, TILE], True)
    other = Q.read_pf_sr(Q.find_pf_sr(den / "planet", "6466_3380_13", design.days[4]), design, ref)
    assert other["same_grid_as_pf_qa"] is False


@pytest.mark.parametrize("make", [
    lambda folder, day: None,
    lambda folder, day: (write_sr(folder / f"{day}.tif"),
                         write_sr(folder / f"{day:%Y_%m_%d}.tif")),
])
def test_no_or_two_pf_sr_files_for_the_example_frame_fail_pf_sr(tmp_path, make):
    design, den, folder, _ = pf_sr_case(tmp_path)
    write_sr(folder / f"{design.days[5]}.tif")
    make(folder, design.days[2])
    with pytest.raises(Q.CheckFailed, match="pf_sr"):
        Q.find_pf_sr(den / "planet", "6466_3380_13", design.days[2])


@pytest.mark.parametrize("make", [
    lambda path: path.write_bytes(b"not a GeoTIFF"),
    lambda path: write_sr(path, count=3),
    lambda path: write_sr(path, size=TILE - 2),
    truncated,
])
def test_an_unusable_pf_sr_file_fails_pf_sr(tmp_path, make):
    design, den, folder, ref = pf_sr_case(tmp_path)
    make(folder / f"{design.days[1]}.tif")
    path = Q.find_pf_sr(den / "planet", "6466_3380_13", design.days[1])
    with pytest.raises(Q.CheckFailed, match="pf_sr"):
        Q.read_pf_sr(path, design, ref)
