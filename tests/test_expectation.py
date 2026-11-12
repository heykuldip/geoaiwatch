"""The expectation printed by analyze.py is the one written in train.py."""
import re
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "posts" / "2026-11-near-infrared" / "scripts"


def _words(text):
    return " ".join(text.split())


def test_printed_expectation_matches_train_py():
    train = (SCRIPTS / "train.py").read_text()
    m = re.search(r"# confusion diagnostic\. Expectation: (.+?phenology\.)", train, re.S)
    recorded = _words(re.sub(r"\n#", " ", m.group(1)))
    analyze = (SCRIPTS / "analyze.py").read_text()
    m = re.search(r'print\("Expectation in train\.py: (.+?phenology\.)"\)', analyze, re.S)
    printed = _words(re.sub(r'\\n"\s*"', " ", m.group(1)))
    assert printed == recorded
