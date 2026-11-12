"""analyze.py reproduces the committed analysis output from the committed run files."""
import subprocess
import sys
from pathlib import Path

import pytest

POST = Path(__file__).resolve().parents[1] / "posts" / "2026-11-near-infrared"


@pytest.mark.parametrize("runs, expected", [("grid_isolated", "analysis.txt"),
                                            ("grid_overlapping", "overlapping/analysis.txt")])
def test_analysis_reproduces(runs, expected):
    d = POST / "runs" / runs
    if len([p for p in d.glob("*.json") if p.name != "audit.json"]) < 12:
        pytest.skip("run files not present")
    out = subprocess.run([sys.executable, str(POST / "scripts" / "analyze.py"), "--runs", str(d)],
                         capture_output=True, text=True, check=True).stdout
    assert out == (POST / "runs" / expected).read_text()
