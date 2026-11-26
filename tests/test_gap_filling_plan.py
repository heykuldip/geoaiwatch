"""The gap-filling post's plan as its code reads it: the frozen file, its amendments, its values,
and the numbers and wording the code takes from its text."""
import json
import re
import shutil

import pytest
from gap_filling_fixtures import SCRIPTS, scripts_on_path

scripts_on_path()
import series_plan as SP  # noqa: E402
import series_qa as Q  # noqa: E402

from geoaiwatch import config  # noqa: E402

PLAN = json.loads(SP.PLAN.read_text())
TEXT = SP.PLAN.read_text()


def test_the_plan_is_the_frozen_one():
    assert SP.sha256(SP.PLAN) == SP.PLAN_SHA256


def test_another_plan_is_refused(tmp_path):
    changed = tmp_path / "plan.json"
    changed.write_text(TEXT.replace('"version": 2', '"version": 3'))
    with pytest.raises(SP.PlanError, match="SHA-256"):
        SP.load(changed, tmp_path / "amendments.json")


def amendments(tmp_path, items, plan_sha256=SP.PLAN_SHA256):
    p = tmp_path / "amendments.json"
    doc = {"amendments": items} if plan_sha256 is None else {"plan_sha256": plan_sha256,
                                                             "amendments": items}
    p.write_text(json.dumps(doc))
    return p


def read_own(path):
    """read_amendments, with the file's own SHA-256 as the pinned one, to test what follows it."""
    return SP.read_amendments(path, pinned=SP.sha256(path))


ONE = {"id": 1, "date": "2026-11-18", "results_seen": "none", "reason": "outputs"}


@pytest.mark.parametrize("items, message", [
    ([], "implements"), ([ONE, dict(ONE, id=2)], "implements"),
    ([dict(ONE, results_seen="")], "lacks"), ([dict(ONE, id=2)], "implements")])
def test_amendments_are_exactly_those_the_code_implements(tmp_path, items, message):
    with pytest.raises(SP.PlanError, match=message):
        read_own(amendments(tmp_path, items))
    assert read_own(amendments(tmp_path, [ONE])) == [ONE]


@pytest.mark.parametrize("plan_sha256", ["0" * 64, None])
def test_amendments_naming_another_plan_or_none_are_refused(tmp_path, plan_sha256):
    with pytest.raises(SP.PlanError, match="frozen plan"):
        read_own(amendments(tmp_path, [ONE], plan_sha256=plan_sha256))


def test_a_missing_amendments_file_is_refused(tmp_path):
    with pytest.raises(SP.PlanError, match="missing"):
        SP.read_amendments(tmp_path / "absent.json")


def test_the_pinned_amendments_are_the_committed_file():
    assert SP.sha256(SP.AMENDMENTS) == SP.AMENDMENTS_SHA256


def test_a_changed_amendment_with_the_same_id_is_refused(tmp_path):
    text = SP.AMENDMENTS.read_text()
    words = "reported with a minus sign, so the maximum past offset is the earliest source"
    assert words in text
    changed = tmp_path / "amendments.json"
    changed.write_text(text.replace(words, "reported as they are"))
    with pytest.raises(SP.PlanError, match="SHA-256"):
        SP.read_amendments(changed)


def test_the_committed_plan_and_amendments_load():
    design, record = SP.load()
    assert record["plan_sha256"] == SP.PLAN_SHA256
    assert [a["id"] for a in record["amendments"]] == list(SP.IMPLEMENTED)
    assert design.n_days == 730


def test_the_design():
    d = SP.design_from(PLAN)
    assert [p.cube for p in d.places] == ["6466_3380_13", "6730_3430_13", "6813_3313_13",
                                           "7026_3201_13"]
    assert [p.name for p in d.places] == ["Renshou", "Fengcheng", "Yangzhou", "Andong"]
    assert (d.n_days, str(d.first_day), str(d.last_day)) == (730, "2018-01-01", "2019-12-31")
    assert len(d.label_dates) == 24 and all(x.day == 1 for x in d.label_dates)
    assert (str(d.label_dates[0]), str(d.label_dates[-1])) == ("2018-01-01", "2019-12-01")
    assert (d.tile, d.quarter, d.bands, d.dtype, d.nodata) == (1024, 512, 9, "int16", -1000)
    assert {w.name: (w.first, w.last, w.days) for w in d.windows} == {
        "nested_15": (15, 1, 15), "nested_30": (30, 1, 30), "nested_60": (60, 1, 60),
        "nested_90": (90, 1, 90), "block_1_30": (30, 1, 30), "block_31_60": (60, 31, 30),
        "block_61_90": (90, 61, 30)}


def test_the_quarters_are_post1s():
    d = SP.design_from(PLAN)
    assert {k: (r, c) for k, r, c in d.quarters} == config.QUADS


def test_the_numbers_come_from_the_plans_text():
    s, lo = SP.COMPLETE_DATES, SP.COMPLETE_CELLS
    assert (f"{s['short']} (2018-02-01 to 2019-12-01; {lo['short']} cells) for nested 15 and 30 "
            f"days and block 1-30, and {s['long']} (2018-04-01 to 2019-12-01; {lo['long']} cells) "
            "for nested 60 and 90 days and blocks 31-60 and 61-90") in TEXT
    assert SP.SHORT_WINDOWS == ("nested_15", "nested_30", "block_1_30")
    f, q = SP.POST1_FRAMES, SP.POST1_QUARTERS
    assert (f"{f['all_1']} frames with every pixel 1, {f['all_100']} with every pixel 100, "
            f"{f['other']} others") in TEXT
    assert f"row by row ({q['fully_actual']} and {q['entirely_filled']})" in TEXT
    c = SP.SAME_DAY_CODES
    assert (f"occurs in {c['frames']} frames, on {c['pixels']:,} pixels "
            f"(900: {c[900]:,}; -900: {c[-900]:,})") in TEXT
    assert c[900] + c[-900] == c["pixels"]
    r = SP.ROWS
    assert f"runs/qa_daily.csv: one row per place, day and quarter ({r['qa_daily.csv']:,})" in TEXT
    assert f"runs/label_frames.csv ({r['label_frames.csv']} rows)" in TEXT
    assert f"runs/label_cells.csv ({r['label_cells.csv']} rows)" in TEXT
    assert f"3 of the 30-day blocks ({r['block_coverage.csv']} rows)" in TEXT


def at(tree: dict, dotted: str):
    for k in dotted.split("."):
        tree = tree[k]
    return tree


def test_the_definitions_are_the_plans_words():
    for key, words in Q.DEFINITIONS.items():
        assert at(PLAN["definitions"], key) == words, key


def test_the_invariant_and_expectation_ids():
    assert tuple(x["id"] for x in PLAN["invariants"]) == SP.INVARIANTS
    assert tuple(x["id"] for x in PLAN["expectations"]) == SP.EXPECTATIONS


def test_layer2_is_named_as_its_source():
    for path in SCRIPTS.glob("*.py"):
        assert not re.search(r"(?i)\b(nearest|closest)\b", path.read_text()), path.name


def test_a_plan_whose_quarters_overlap_is_refused():
    plan = json.loads(TEXT)
    plan["data"]["quarters"]["origin_row_col"]["SE"] = [500, 512]
    with pytest.raises(SP.PlanError, match="partition"):
        SP.design_from(plan)


def test_a_plan_with_another_invariant_list_is_refused():
    plan = json.loads(TEXT)
    plan["invariants"] = plan["invariants"][:-1]
    with pytest.raises(SP.PlanError, match="invariants"):
        SP.design_from(plan)


def test_a_copy_of_the_plan_loads_with_its_amendments(tmp_path):
    shutil.copy(SP.PLAN, tmp_path / "plan.json")
    shutil.copy(SP.AMENDMENTS, tmp_path / "amendments.json")
    design, record = SP.load(tmp_path / "plan.json", tmp_path / "amendments.json")
    assert record["amendments_sha256"] == SP.AMENDMENTS_SHA256
    assert len(design.places) == 4
