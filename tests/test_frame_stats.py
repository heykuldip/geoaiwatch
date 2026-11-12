"""Regression test for scripts/frame_stats.py: the frame-level numbers behind the Post 1 figures."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POST = ROOT / "posts" / "2026-11-near-infrared"


def run(capsys, monkeypatch, tables):
    need = [tables / "cell_level.csv", POST / "artifacts" / "cell_qa_status.parquet"]
    if not all(p.exists() for p in need):
        pytest.skip("committed result tables missing")
    monkeypatch.syspath_prepend(str(POST / "scripts"))
    sys.modules.pop("frame_stats", None)
    importlib.import_module("frame_stats").main(tables)
    return capsys.readouterr().out


def test_frame_stats(capsys, monkeypatch):
    out = run(capsys, monkeypatch, POST / "runs")
    assert "ΔmIoU mean +0.101, median +0.105" in out
    assert "Δ +0.136 (seeds +0.137, +0.146, +0.126); 82% of cells gain less; 1.3x the median" in out
    assert "Δ +0.473 (seeds +0.473, +0.460, +0.487); 65% of cells gain less" in out
    assert "Δ +0.182 (seeds +0.182, +0.158, +0.204); 98% of cells gain less; 1.7x the median" in out
    assert "Δ +0.691 (seeds +0.713, +0.553, +0.807); 96% of cells gain less" in out


def test_frame_stats_earlier_runs(capsys, monkeypatch):
    out = run(capsys, monkeypatch, POST / "runs" / "overlapping")
    assert "ΔmIoU mean +0.063, median +0.062" in out
    assert "Δ +0.119 (seeds +0.101, +0.139, +0.116); 92% of cells gain less; 1.9x the median" in out
