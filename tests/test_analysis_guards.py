"""analyze.py accepts only the complete, uncapped twelve-run grid on the frozen settings."""
import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "posts" / "2026-11-near-infrared" / "scripts"


@pytest.fixture
def analyze(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    sys.modules.pop("analyze", None)
    return importlib.import_module("analyze")


def checkpoints(A, n=None, epochs=None):
    cells = sorted(A.frozen_cells())[:n]
    per_cell = [{"cube": c, "date": d} for c, d in cells]
    return {ep: {"arms": {arm: {"per_cell": per_cell} for arm in ("rgbn", "rgb0")}}
            for ep in (epochs or A.EPOCHS)}


def runs(A, legacy=False, **change):
    hist = [{"epoch": e} if legacy else {"epoch": e, "batches": A.BATCHES_PER_EPOCH}
            for e in range(1, 121)]
    caps = {} if legacy else {"max_batches": None, "max_frames": None}
    out = [dict(A.DESIGN, fold=f, seed=s, smoke=False, classes=["a"], history=hist,
                checkpoints=checkpoints(A), **caps) for f in A.FOLDS for s in A.SEEDS]
    out[0] = dict(out[0], **change)
    return out


def write(tmp_path, rs, audit=None):
    for i, r in enumerate(rs):
        (tmp_path / f"{i:02d}.json").write_text(json.dumps(r))
    if audit is not None:
        (tmp_path / "audit.json").write_text(json.dumps({"runs": audit}))
    return tmp_path


def test_the_frozen_grid_passes(analyze, tmp_path):
    assert len(analyze.load_runs(write(tmp_path, runs(analyze)))) == 12


@pytest.mark.parametrize("change, message", [
    ({"seed": 1}, "fold-seed pairs"),
    ({"epochs": 80, "primary_eval_epoch": 80}, "departs from the frozen design"),
    ({"lr": 1e-3}, "departs from the frozen design"),
    ({"max_batches": 1}, "was capped"),
    ({"history": [{"epoch": e, "batches": 1} for e in range(1, 121)]}, "batches per epoch"),
    ({"history": [{"epoch": e, "batches": 48} for e in range(1, 81)]}, "epochs 1 to 120"),
])
def test_departures_are_refused(analyze, tmp_path, change, message):
    with pytest.raises(RuntimeError, match=message):
        analyze.load_runs(write(tmp_path, runs(analyze, **change)))


def test_a_missing_run_is_refused(analyze, tmp_path):
    with pytest.raises(RuntimeError, match="fold-seed pairs"):
        analyze.load_runs(write(tmp_path, runs(analyze)[1:]))


def test_a_missing_checkpoint_or_cell_is_refused(analyze, tmp_path):
    three = checkpoints(analyze, epochs=analyze.EPOCHS[:3])
    with pytest.raises(RuntimeError, match="checkpoints"):
        analyze.load_runs(write(tmp_path, runs(analyze, checkpoints=three)))
    with pytest.raises(RuntimeError, match="frozen cells"):
        analyze.load_runs(write(tmp_path, runs(analyze, checkpoints=checkpoints(analyze, n=95))))


def entry(batches=(48,)):
    return {"logs": ["grid-1_0.out"], "sha256": ["0" * 64], "epochs": 120, "batches": list(batches)}


def test_runs_without_batch_counts_need_an_audit(analyze, tmp_path):
    rs = runs(analyze, legacy=True)
    with pytest.raises(RuntimeError, match="audit.json"):
        analyze.load_runs(write(tmp_path, rs))
    ok = {f"{r['fold']}_s{r['seed']}": entry() for r in rs}
    assert len(analyze.load_runs(write(tmp_path, rs, audit=ok))) == 12
    with pytest.raises(RuntimeError, match="batches per epoch"):
        analyze.load_runs(write(tmp_path, rs, audit=dict(ok, NE_s0=entry((1, 48)))))


def test_a_declared_cap_is_refused_even_with_an_audit(analyze, tmp_path):
    rs = runs(analyze, legacy=True, max_batches=1)
    ok = {f"{r['fold']}_s{r['seed']}": entry() for r in rs}
    with pytest.raises(RuntimeError, match="was capped"):
        analyze.load_runs(write(tmp_path, rs, audit=ok))


def test_an_audit_entry_needs_its_logs_and_checksums(analyze, tmp_path):
    rs = runs(analyze, legacy=True)
    ok = {f"{r['fold']}_s{r['seed']}": entry() for r in rs}
    for bad in ({"epochs": 120, "batches": [48]}, dict(entry(), sha256=["not-a-hash"])):
        with pytest.raises(RuntimeError, match="SHA-256"):
            analyze.load_runs(write(tmp_path, rs, audit=dict(ok, NE_s0=bad)))


def test_a_missing_fold_at_an_epoch_is_refused(analyze):
    cells = sorted(analyze.frozen_cells())
    rows = [{"fold": f, "seed": s, "epoch": e, "cube": c, "date": d, "d_miou": 0.0}
            for f in analyze.FOLDS for s in analyze.SEEDS for e in (20, 40, 80, 120)
            for c, d in cells if not (f == "NE" and e == 120)]
    with pytest.raises(AssertionError, match="run-epochs"):
        analyze.cell_means(pd.DataFrame(rows))
