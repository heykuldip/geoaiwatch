"""The gap-filling post's run, end to end, on a synthetic dataset with the plan's calendar:
every table against a slow, direct reference computed from the whole series at once; and the
run's guards, each shown to stop it with nothing written."""
import csv
import hashlib
import json
import subprocess
from collections import Counter
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
from gap_filling_fixtures import (TIE, TILE, ZONES, expected_for, make_series, qa_path,
                                  scripts_on_path, small_design, write_dataset, write_post1_table,
                                  write_qa)

scripts_on_path()
import series_plan as SP  # noqa: E402
import series_run as R  # noqa: E402

PLAN = json.loads(SP.PLAN.read_text())
RECORD = {"plan": PLAN, "plan_sha256": SP.PLAN_SHA256, "amendments": [{"id": 1}],
          "amendments_sha256": None}
SOURCES = ("same_day_0", "same_day_900", "same_day_minus900", "past", "future")


def quiet(_):
    pass


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def s(x):
    if x is None:
        return ""
    if isinstance(x, Fraction):
        return f"{float(x):.6f}"
    return str(int(x))


def q(values, k):
    return int(np.quantile(np.asarray(values), k / 10, method="inverted_cdf"))


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    design = small_design()
    series = make_series(design, seed=0)
    root = tmp_path_factory.mktemp("gap_filling")
    write_dataset(root / "den", design, series)
    write_post1_table(root / "post1.parquet", design, series)
    expected = expected_for(design, series)
    audit = R.run(design, RECORD, root / "den", root / "runs", root / "post1.parquet", expected,
                  log=quiet)
    tables = {name: read(root / "runs" / name) for name in R.TABLES}
    return SimpleNamespace(design=design, series=series, root=root, audit=audit, tables=tables,
                           expected=expected)


def classes(a):
    l1, l2, l3 = a[:, 0], a[:, 1], a[:, 2]
    nod = (l1 == -1000) | (l2 == -1000) | (l3 == -1000)
    return {"nodata": nod, "fully_actual": ~nod & (l1 == 1),
            "filled": ~nod & (l1 >= 2) & (l1 <= 100), "entirely_filled": ~nod & (l1 == 100),
            "blended": ~nod & (l1 >= 2) & (l1 <= 99)}


def quarters(design):
    n = design.quarter
    return {k: (slice(r, r + n), slice(c, c + n)) for k, r, c in design.quarters}


def test_every_output_is_written(syn):
    names = {p.name for p in (syn.root / "runs").iterdir()}
    assert names == set(R.OUTPUTS)
    assert set(syn.audit["checks"].values()) == {"passed"}
    assert len(syn.tables["qa_daily.csv"]) == 11_680
    assert (len(syn.tables["label_frames.csv"]), len(syn.tables["label_cells.csv"])) == (96, 384)
    assert len(syn.tables["block_coverage.csv"]) == 336
    assert len(syn.tables["windows.csv"]) == 4 * 24 * 4 * 7


def test_daily_counts(syn):
    rows = {(r["cube"], r["date"], r["quarter"]): r for r in syn.tables["qa_daily.csv"]}
    codes = (("no_planetscope", -999), ("clear", 1), ("code2", 2), ("code3", 3), ("code4", 4),
             ("code5", 5), ("code6", 6), ("misaligned", 7))
    for cube, a in syn.series.items():
        c = classes(a)
        for name, sl in quarters(syn.design).items():
            sub = {k: v[(slice(None),) + sl].sum(axis=(1, 2)) for k, v in c.items()}
            l3 = a[(slice(None), 2) + sl]
            for t, day in enumerate(syn.design.days):
                r = rows[(cube, str(day), name)]
                for k, v in sub.items():
                    assert r[f"n_{k}"] == s(v[t]), (cube, day, name, k)
                for kind in ("entirely_filled", "blended"):
                    m = c[kind][(t,) + sl]
                    for status, code in codes:
                        assert r[f"{kind}_l3_{status}"] == s(((l3[t] == code) & m).sum())


def runs(flags):
    """{censoring: Counter of lengths} for the runs of True in a 1-D series."""
    out = {k: Counter() for k in ("none", "start", "end", "both")}
    n, i = len(flags), 0
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j < n and flags[j]:
            j += 1
        kind = {(True, True): "both", (True, False): "start", (False, True): "end",
                (False, False): "none"}[(i == 0, j == n)]
        out[kind][j - i] += 1
        i = j
    return out


def test_gaps(syn):
    lengths = Counter()
    for r in syn.tables["gap_lengths.csv"]:
        lengths[(r["cube"], r["level"], r["censored"], int(r["length"]))] += int(r["n_gaps"])
        assert int(r["n_days"]) == int(r["n_gaps"]) * int(r["length"])
    summary = {(r["cube"], r["weight"]): r for r in syn.tables["gaps.csv"]}
    for cube, a in syn.series.items():
        fa = classes(a)["fully_actual"]
        want = {k: Counter() for k in ("none", "start", "end", "both")}
        for r in range(TILE):
            for c in range(TILE):
                for k, v in runs(~fa[:, r, c]).items():
                    want[k].update(v)
        none = runs(fa.sum(axis=(1, 2)) == 0)
        for level, hist in (("pixel", want), ("tile_none", none)):
            for k, counts in hist.items():
                for length, n in counts.items():
                    assert lengths.pop((cube, level, k, length)) == n
        u = want["none"]
        by_days = np.repeat(list(u), [n * L for L, n in u.items()])
        by_gaps = np.repeat(list(u), list(u.values()))
        tile = np.repeat(list(none["none"]), list(none["none"].values()))
        for weight, sample, censored in (("pixel_days", by_days, want), ("gaps", by_gaps, want),
                                         ("runs", tile, none)):
            r = summary[(cube, weight)]
            assert r["n_uncensored"] == s(len(sample))
            want = (s(q(sample, 5)), s(q(sample, 9)), s(sample.max())) if len(sample) else \
                ("", "", "")
            assert (r["median"], r["p90"], r["max"]) == want
            for k in ("start", "end", "both"):
                n = sum(n * (L if weight == "pixel_days" else 1) for L, n in censored[k].items())
                assert r[f"n_censored_{k}"] == s(n)
    assert not lengths


def label_reference(a, sl=(slice(None), slice(None))):
    """Estimand 3 for one frame or quarter, directly."""
    l1, l2 = a[0][sl], a[1][sl]
    nod = ((a[0] == -1000) | (a[1] == -1000) | (a[2] == -1000))[sl]
    filled = ~nod & (l1 >= 2) & (l1 <= 100)
    src = {"same_day_0": filled & (l2 == 0), "same_day_900": filled & (l2 == 900),
           "same_day_minus900": filled & (l2 == -900), "past": filled & (l2 < 0) & (l2 > -900),
           "future": filled & (l2 > 0) & (l2 < 900)}
    out = {"n_pixels": l1.size, "n_nodata": int(nod.sum()),
           "n_fully_actual": int((~nod & (l1 == 1)).sum()), "n_filled": int(filled.sum()),
           "n_entirely_filled": int((filled & (l1 == 100)).sum()),
           "n_blended": int((filled & (l1 < 100)).sum())}
    out.update({f"n_l2_{k}": int(v.sum()) for k, v in src.items()})
    for side, sign in (("past", -1), ("future", 1)):
        d = np.abs(l2[src[side]])
        out[f"{side}_offset_median"] = sign * q(d, 5) if d.size else None
        out[f"{side}_offset_p90"] = sign * q(d, 9) if d.size else None
        out[f"{side}_offset_max"] = sign * int(d.max()) if d.size else None
    out.update(l1_all_1=bool((l1 == 1).all()), l1_all_100=bool((l1 == 100).all()),
               n_l2_900_all=int((l2 == 900).sum()), n_l2_minus900_all=int((l2 == -900).sum()))
    return out, Counter(l2[filled].tolist())


def compare_label(row, ref):
    for k, v in ref.items():
        want = ("true" if v else "false") if isinstance(v, bool) else s(v)
        assert row[k] == want, k
    f = ref["n_filled"]
    assert row["share_filled"] == s(Fraction(f, ref["n_pixels"]))
    for k in SOURCES:
        assert row[f"share_l2_{k}"] == (s(Fraction(ref[f"n_l2_{k}"], f)) if f else "")
    assert row["share_future_of_pixels"] == s(Fraction(ref["n_l2_future"], ref["n_pixels"]))


def test_label_dates(syn):
    frames = {(r["cube"], r["date"]): r for r in syn.tables["label_frames.csv"]}
    cells = {(r["cube"], r["date"], r["quarter"]): r for r in syn.tables["label_cells.csv"]}
    codes = Counter()
    for r in syn.tables["label_codes.csv"]:
        codes[(r["cube"], r["date"], r["quarter"], int(r["l2"]))] += int(r["n_filled"])
    pooled, shares, hist = Counter(), [], Counter()
    for cube, a in syn.series.items():
        for day in syn.design.label_dates:
            x = a[syn.design.index(day)]
            ref, h = label_reference(x)
            compare_label(frames[(cube, str(day))], ref)
            pooled.update({k: v for k, v in ref.items() if k.startswith("n_")
                           and not k.endswith("_all")})
            hist.update(h)
            shares.append((Fraction(ref["n_l2_future"], ref["n_pixels"]), day, int(cube[:4]),
                           cube, ref))
            for name, sl in quarters(syn.design).items():
                ref_q, h_q = label_reference(x, sl)
                compare_label(cells[(cube, str(day), name)], ref_q)
                for code, n in h_q.items():
                    assert codes.pop((cube, str(day), name, code)) == n
    assert not codes
    summary = syn.tables["label_summary.csv"][0]
    for k in ("n_pixels", "n_fully_actual", "n_filled") + tuple(f"n_l2_{k}" for k in SOURCES):
        assert summary[k] == s(pooled[k]), k
    assert summary["share_l2_future"] == s(Fraction(pooled["n_l2_future"], pooled["n_filled"]))
    dw = [Fraction(r["n_l2_future"], r["n_filled"]) for *_, r in shares if r["n_filled"]]
    assert summary["share_dw_l2_future"] == s(sum(dw) / len(dw))
    past_codes = [c for c in hist if -900 < c < 0]
    past = np.repeat([-c for c in past_codes], [hist[c] for c in past_codes])
    assert summary["past_offset_median"] == s(-q(past, 5))
    assert summary["past_offset_max"] == s(-past.max())
    best = sorted([x for x in shares if x[0] > 0], key=lambda x: (-x[0], x[1], x[2]))[0]
    assert (best[3], best[1]) == TIE[0]
    assert (summary["example_cube"], summary["example_date"]) == (TIE[0][0], str(TIE[0][1]))
    ranked = sorted(x[0] for x in shares)
    assert summary["median_share_future_of_pixels"] == s(ranked[47])


def test_windows_and_blocks(syn):
    d = syn.design
    rows = {(r["cube"], r["date"], r["quarter"], r["window"]): r for r in syn.tables["windows.csv"]}
    hist = Counter()
    for r in syn.tables["window_days.csv"]:
        hist[(r["cube"], r["date"], r["quarter"], r["window"], int(r["fully_actual_days"]))] += \
            int(r["n_pixels"])
    blocks = {(r["cube"], r["date"], r["quarter"]): r for r in syn.tables["block_coverage.csv"]}
    for cube, a in syn.series.items():
        fa = classes(a)["fully_actual"]
        for day in d.label_dates:
            t = d.index(day)
            for name, sl in quarters(d).items():
                covered = {}
                for w in d.windows:
                    r = rows[(cube, str(day), name, w.name)]
                    if t - w.first < 0:
                        assert r["complete"] == "false" and r["n_covered"] == ""
                        continue
                    sub = fa[(slice(t - w.first, t - w.last + 1),) + sl]
                    count = sub.sum(axis=0).ravel()
                    covered[w.name] = count > 0
                    assert r["complete"] == "true"
                    assert r["n_covered"] == s((count > 0).sum())
                    want = [s(q(count, k)) for k in (1, 5, 9)]
                    assert [r[f"days_p{k}0"] for k in (1, 5, 9)] == want
                    assert r["n_days_any"] == s(sub.reshape(len(sub), -1).any(axis=1).sum())
                    for k, n in Counter(count.tolist()).items():
                        assert hist.pop((cube, str(day), name, w.name, k)) == n
                if "block_61_90" in covered:
                    k = sum(covered[b].astype(int) for b in ("block_1_30", "block_31_60",
                                                             "block_61_90"))
                    b = blocks.pop((cube, str(day), name))
                    assert [b[f"n_blocks_{i}"] for i in range(4)] == \
                        [s(v) for v in np.bincount(k, minlength=4)]
    assert not hist and not blocks


def test_windows_summary(syn):
    d = syn.design
    windows = syn.tables["windows.csv"]
    summary = {(r["cube"], r["window"], r["dates"]): r for r in syn.tables["windows_summary.csv"]}
    common = [x for x in d.label_dates if x >= d.label_dates[3]]
    for cube, a in syn.series.items():
        fa = classes(a)["fully_actual"]
        for w in d.windows:
            own = [x for x in d.label_dates if d.index(x) - w.first >= 0]
            for label, dates in (("own", own), ("common", common)):
                r = summary[(cube, w.name, label)]
                counts = [fa[(slice(d.index(x) - w.first, d.index(x) - w.last + 1),) + sl]
                          .sum(axis=0).ravel() for x in dates for sl in quarters(d).values()]
                assert r["n_cells"] == s(len(counts))
                every = np.concatenate(counts)
                assert r["share_covered"] == s(Fraction(int((every > 0).sum()), every.size))
                cell_shares = sorted(Fraction(int((c > 0).sum()), c.size) for c in counts)
                assert r["share_covered_median"] == s(cell_shares[-(-5 * len(counts) // 10) - 1])
                assert [r[f"days_p{k}0"] for k in (1, 5, 9)] == [s(q(every, k)) for k in (1, 5, 9)]
                cells = [x for x in windows if x["cube"] == cube and x["window"] == w.name
                         and x["date"] in {str(y) for y in dates}]
                assert r["n_days_any_median"] == s(q([int(x["n_days_any"]) for x in cells], 5))


def test_monthly(syn):
    d = syn.design
    rows = {(r["cube"], r["period"]): r for r in syn.tables["qa_monthly.csv"]}
    for cube, a in syn.series.items():
        c = classes(a)
        l3 = a[:, 2]
        tile_fa = c["fully_actual"].sum(axis=(1, 2))
        for month in ("2018-02", "2019-07"):
            idx = [t for t, x in enumerate(d.days) if f"{x:%Y-%m}" == month]
            r = rows[(cube, month)]
            n = c["fully_actual"][idx].sum()
            assert r["share_fully_actual"] == s(Fraction(int(n), len(idx) * TILE * TILE))
            assert r["days_none"] == s((tile_fa[idx] == 0).sum())
            assert r["days_all"] == s((tile_fa[idx] == TILE * TILE).sum())
            for kind in ("entirely_filled", "blended"):
                m = c[kind][idx]
                hit = m & (l3[idx] == -999)
                assert r[f"share_{kind}_l3_no_planetscope"] == \
                    s(Fraction(int(hit.sum()), int(m.sum())))
                per_day = [Fraction(int(h.sum()), int(k.sum())) for h, k in zip(hit, m) if k.sum()]
                assert r[f"share_dw_{kind}_l3_no_planetscope"] == s(sum(per_day) / len(per_day))
                haze = m & (l3[idx] >= 2) & (l3[idx] <= 6)
                assert r[f"n_{kind}_l3_cloud_shadow_haze"] == s(haze.sum())


def test_expectations(syn):
    ex = {e["id"]: e for e in json.loads((syn.root / "runs" / "expectations.json")
                                         .read_text())["expectations"]}
    assert list(ex) == list(SP.EXPECTATIONS)
    nodata = sum(int(classes(a)["nodata"].sum()) for a in syn.series.values())
    assert (ex["EX1"]["observed"]["nodata_pixel_days"], ex["EX1"]["met"]) == (nodata, nodata == 0)
    fa = sum(int(classes(a)["fully_actual"].sum()) for a in syn.series.values())
    total = 4 * 730 * TILE * TILE
    assert ex["EX2"]["observed"]["fully_actual"] == fa
    assert ex["EX2"]["met"] == (Fraction(15, 100) <= Fraction(fa, total) <= Fraction(40, 100))
    some = {c: int(((classes(a)["fully_actual"].sum(axis=(1, 2)) > 0)
                    & (classes(a)["fully_actual"].sum(axis=(1, 2)) < TILE * TILE)).sum())
            for c, a in syn.series.items()}
    assert ex["EX4"]["observed"]["days_some"] == some
    assert ex["EX4"]["met"] == all(3 * v < 730 for v in some.values())
    for e in ex.values():
        assert e["statement"] == next(x["statement"] for x in PLAN["expectations"]
                                      if x["id"] == e["id"])


def test_inputs_and_audit(syn):
    lines = (syn.root / "runs" / "inputs.sha256").read_text().splitlines()
    assert len(lines) == 4 * 730 + 1
    for line in lines[:3] + lines[-2:]:
        digest, path = line.split("  ")
        assert not path.startswith("/")
        assert hashlib.sha256((syn.root / "den" / path).read_bytes()).hexdigest() == digest
    cube, day = TIE[0]
    assert lines[-1].endswith(f"{cube}/PF-SR/{day}.tif")
    a = syn.audit
    assert a["plan"]["sha256"] == SP.PLAN_SHA256
    assert a["inputs"]["pf_qa_files"] == {c: 730 for c in ZONES}
    assert len(a["inputs"]["metadata"]) == 8
    assert a["example_frame"]["cube"] == cube
    check = a["example_frame"]["pf_sr_check"]
    assert (check["bands"], check["pixels"], check["same_grid_as_pf_qa"]) == (4, [TILE, TILE], True)
    assert lines[-1].startswith(check["sha256"])
    assert (a["code"]["commit"], a["code"]["git"]) == (None, "not used")
    assert set(a["code"]["files"]) == {"posts/2026-11-gap-filling/scripts/series_plan.py",
                                       "posts/2026-11-gap-filling/scripts/series_qa.py",
                                       "posts/2026-11-gap-filling/scripts/series_run.py",
                                       "src/geoaiwatch/config.py"}
    for name, digest in a["code"]["files"].items():
        assert hashlib.sha256((R.REPO / name).read_bytes()).hexdigest() == digest
    fully = Counter()
    for c, arr in syn.series.items():
        m = classes(arr)["fully_actual"]
        fully.update(arr[:, 1][m].tolist())
    got = Counter()
    for c in ZONES:
        got.update({int(k): v for k, v in a["diagnostics"][c]["l2_codes_on_fully_actual"].items()})
    assert got == fully


def test_a_check_failing_after_the_pass_writes_nothing(syn, tmp_path):
    wrong = R.Expected(post1_frames=dict(syn.expected.post1_frames, other=0),
                       post1_quarters=syn.expected.post1_quarters,
                       same_day_codes=syn.expected.same_day_codes)
    with pytest.raises(R.Q.CheckFailed, match="I9"):
        R.run(syn.design, RECORD, syn.root / "den", tmp_path / "runs", syn.root / "post1.parquet",
              wrong, log=quiet)
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())


def test_tables_that_read_back_differently_write_nothing(syn, tmp_path, monkeypatch):
    real = R.read_csv

    def altered(path):
        lines = real(path)
        if path.name == "gaps.csv":
            lines[1][-1] = "0"
        return lines

    monkeypatch.setattr(R, "read_csv", altered)
    with pytest.raises(R.Q.CheckFailed, match="tables"):
        R.run(syn.design, RECORD, syn.root / "den", tmp_path / "runs", syn.root / "post1.parquet",
              syn.expected, log=quiet)
    assert list((tmp_path / "runs").iterdir()) == []


def test_a_change_to_the_code_during_the_run_writes_nothing(syn, tmp_path):
    states = iter([{"commit": "a", "clean": True}, {"commit": "b", "clean": True}])
    with pytest.raises(R.Q.CheckFailed, match="commit"):
        R.run(syn.design, RECORD, syn.root / "den", tmp_path / "runs", syn.root / "post1.parquet",
              syn.expected, git=lambda: next(states), log=quiet)
    assert list((tmp_path / "runs").iterdir()) == []


@pytest.mark.parametrize("state", [{"commit": "a", "clean": False},
                                   {"commit": None, "clean": None},
                                   {"commit": "", "clean": True}])
def test_code_not_in_a_clean_git_checkout_is_refused(syn, tmp_path, state):
    with pytest.raises(R.Q.CheckFailed, match="commit"):
        R.run(syn.design, RECORD, syn.root / "den", tmp_path / "a", syn.root / "post1.parquet",
              syn.expected, git=lambda: state, log=quiet)
    assert not (tmp_path / "a").exists()


def test_earlier_outputs_are_refused(syn, tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "audit.json").write_text("{}")
    with pytest.raises(R.Q.CheckFailed, match="outputs"):
        R.run(syn.design, RECORD, syn.root / "den", tmp_path / "b", syn.root / "post1.parquet",
              syn.expected, log=quiet)


def test_wrong_window_numbers_fail_i8_before_any_pixel(syn, tmp_path):
    wrong = R.Expected(complete_dates={"short": 23, "long": 22})
    with pytest.raises(R.Q.CheckFailed, match="I8"):
        R.run(syn.design, RECORD, syn.root / "den", tmp_path / "runs", syn.root / "post1.parquet",
              wrong, log=quiet)


def test_no_future_source_fails_i11():
    design = small_design()
    series = make_series(design, seed=1, future=False)
    results = [SimpleNamespace(cube=c, frames=[
        {"date": x, "n_pixels": TILE * TILE,
         "n_l2_future": int(((a[design.index(x), 0] >= 2) & (a[design.index(x), 1] >= 1)
                             & (a[design.index(x), 1] <= 899)).sum())}
        for x in design.label_dates]) for c, a in series.items()]
    assert all(f["n_l2_future"] == 0 for r in results for f in r.frames)
    with pytest.raises(R.Q.CheckFailed, match="I11"):
        R.example_frame(results)


def test_check_inputs_records_no_counts(syn, tmp_path):
    seen = []
    report = R.check_inputs(syn.design, RECORD, syn.root / "den", tmp_path, log=seen.append)
    assert report["result"] == "passed"
    assert {k: v for k, v in report["checks"].items()} == {
        "I8": "passed", "I1": "passed", "I2": "passed (headers)", "I3": "passed"}
    assert set(report) == {"plan_sha256", "checks", "metadata", "grids", "files",
                           "metadata_beyond_image_structure", "result", "seconds"}
    assert all(line.split(":")[1].startswith(" I3 passed on 730 files") for line in seen)
    assert json.loads((tmp_path / "input_check.json").read_text())["result"] == "passed"


def test_check_inputs_names_codes_out_of_range(tmp_path):
    import dataclasses
    design = small_design()
    design = dataclasses.replace(design, n_days=10, label_dates=(design.first_day,))
    den = tmp_path / "den"
    layers = np.ones((3, TILE, TILE), np.int16)
    for p in design.places:
        for day in design.days:
            write_qa(qa_path(den, p.cube, day), layers, ZONES[p.cube][1])
        sr = qa_path(den, p.cube, design.first_day).parent.parent / "PF-SR"
        sr.mkdir(parents=True)
        write_qa(sr / f"{design.first_day}.tif", layers, ZONES[p.cube][1])
    bad = layers.copy()
    bad[0, 2, 2] = 0
    write_qa(qa_path(den, "6730_3430_13", design.days[5]), bad, ZONES["6730_3430_13"][1])
    none = {"short": 0, "long": 0}
    report = R.check_inputs(design, RECORD, den, tmp_path / "out",
                            R.Expected(complete_dates=none, complete_cells=none), log=quiet)
    assert report["result"] == "I3 failed"
    assert report["problems"] == [f"6730_3430_13 {design.days[5]}.tif: "
                                  "L1 codes outside its range: [0]"]


def test_git_state(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@localhost")
    git("config", "user.name", "t")
    (tmp_path / "a.txt").write_text("1")
    git("add", "a.txt")
    git("commit", "-q", "-m", "a")
    assert R.git_state(tmp_path)["clean"] is True
    (tmp_path / "b.txt").write_text("untracked")
    assert R.git_state(tmp_path)["clean"] is True
    (tmp_path / "a.txt").write_text("2")
    assert R.git_state(tmp_path)["clean"] is False
    assert R.git_state(tmp_path / "missing")["commit"] is None


def test_the_run_refuses_a_plan_that_fails_to_load(monkeypatch):
    def fail():
        raise SP.PlanError("no")

    monkeypatch.setattr(SP, "load", fail)
    assert R.main([]) == 1


def test_post1s_table():
    if not R.POST1_CELLS.exists():
        pytest.skip("Post 1's cell_qa_status.parquet is absent")
    cells = R.read_post1_cells(R.POST1_CELLS)
    design = SP.design_from(PLAN)
    assert len(cells) == 384
    assert {k[0] for k in cells} == {p.cube for p in design.places}
    assert {k[1] for k in cells} == {str(x) for x in design.label_dates}
    assert {k[2] for k in cells} == {"NW", "NE", "SW", "SE"}
    assert (sum(a for a, _ in cells.values()), sum(b for _, b in cells.values())) == (70, 237)


# the checks across places and tables, each made to fail

def tiny_results():
    d1, d2 = SP.design_from(PLAN).label_dates[:2]
    cells = [{"date": d, "quarter": q, "l1_all_1": d == d1, "l1_all_100": False}
             for d in (d1, d2) for q in ("NW", "NE", "SW", "SE")]
    frames = [{"date": d1, "l1_all_1": True, "l1_all_100": False, "n_l2_900_all": 3,
               "n_l2_minus900_all": 0},
              {"date": d2, "l1_all_1": False, "l1_all_100": False, "n_l2_900_all": 0,
               "n_l2_minus900_all": 2}]
    post1 = {("6466_3380_13", str(c["date"]), c["quarter"]): (c["l1_all_1"], c["l1_all_100"])
             for c in cells}
    expected = R.Expected(post1_frames={"all_1": 1, "all_100": 0, "other": 1},
                          post1_quarters={"fully_actual": 4, "entirely_filled": 0},
                          same_day_codes={"frames": 2, 900: 3, -900: 2, "pixels": 5})
    return [SimpleNamespace(cube="6466_3380_13", frames=frames, cells=cells)], post1, expected


def test_a_flag_flipped_in_post1s_table_fails_i9():
    results, post1, expected = tiny_results()
    R.check_post1(results, post1, expected)
    key = next(iter(post1))
    post1[key] = (not post1[key][0], post1[key][1])
    with pytest.raises(R.Q.CheckFailed, match="I9"):
        R.check_post1(results, post1, expected)


def test_one_same_day_pixel_changed_fails_i10():
    results, _, expected = tiny_results()
    R.check_same_day_codes(results, expected)
    results[0].frames[1]["n_l2_minus900_all"] += 1
    with pytest.raises(R.Q.CheckFailed, match="I10"):
        R.check_same_day_codes(results, expected)


def test_a_day_counted_in_two_classes_fails_i6(monkeypatch):
    design = SP.design_from(PLAN)
    results = [SimpleNamespace(cube="6466_3380_13", tile_fully_actual=[0] * design.n_days)]
    R.check_day_classes(results, design)
    real = R.day_classes
    monkeypatch.setattr(R, "day_classes", lambda fa, d: real(fa, d) + ["some"])
    with pytest.raises(R.Q.CheckFailed, match="I6"):
        R.check_day_classes(results, design)


def test_sources_that_miss_a_filled_pixel_fail_i4(monkeypatch):
    design = small_design()
    l1 = np.full((TILE, TILE), 100, np.int16)
    l2 = np.full((TILE, TILE), 5, np.int16)
    l3 = np.ones((TILE, TILE), np.int16)
    c = R.Q.classify(l1, l2, l3, -1000)
    qs = R.Q.quarter_slices(design)
    counts = R.Q.day_counts(c, l3, qs)
    R.Q.label_frame(design.label_dates[0], l1, l2, c, counts, qs)
    real = R.Q.sources

    def missing(l2, filled):
        out = real(l2, filled)
        out["future"] = out["future"].copy()
        out["future"][0, 0] = False
        return out

    monkeypatch.setattr(R.Q, "sources", missing)
    with pytest.raises(R.Q.CheckFailed, match="I4"):
        R.Q.label_frame(design.label_dates[0], l1, l2, c, counts, qs)


def test_block_coverage_that_disagrees_with_windows_fails_blocks_sum():
    head = ["cube", "date", "quarter", "window", "n_covered"]
    key = ["6466_3380_13", "2018-04-01", "NW"]
    windows = [head] + [key + [w, n] for w, n in (("block_1_30", "5"), ("block_31_60", "4"),
                                                   ("block_61_90", "3"), ("nested_90", "7"))]
    blocks = [["cube", "date", "quarter", "n_blocks_0", "n_blocks_1", "n_blocks_2",
               "n_blocks_3"], key + ["2", "4", "1", "2"]]
    R.check_blocks_sum(windows, blocks)
    blocks[1][-1] = "1"
    with pytest.raises(R.Q.CheckFailed, match="blocks_sum"):
        R.check_blocks_sum(windows, blocks)


def test_a_gap_left_out_of_the_histograms_fails_gap_days():
    hist = {k: np.zeros(4, np.int64) for k in R.Q.CENSORING}
    hist["none"][2] = 1
    daily = [{"n_pixels": 1, "n_fully_actual": 1}, {"n_pixels": 1, "n_fully_actual": 0},
             {"n_pixels": 1, "n_fully_actual": 0}]
    result = SimpleNamespace(cube="6466_3380_13", gaps=hist, daily=daily)
    R.check_gap_days([result])
    hist["none"][2] = 0
    with pytest.raises(R.Q.CheckFailed, match="gap_days"):
        R.check_gap_days([result])


def test_a_missing_pf_sr_for_the_example_frame_fails_after_the_pass(syn, tmp_path):
    cube, day = TIE[0]
    sr = qa_path(syn.root / "den", cube, day).parent.parent / "PF-SR" / f"{day}.tif"
    aside = sr.with_name("aside.bin")
    sr.rename(aside)
    try:
        with pytest.raises(R.Q.CheckFailed, match="pf_sr"):
            R.run(syn.design, RECORD, syn.root / "den", tmp_path / "runs",
                  syn.root / "post1.parquet", syn.expected, log=quiet)
    finally:
        aside.rename(sr)
    assert not (tmp_path / "runs").exists()
