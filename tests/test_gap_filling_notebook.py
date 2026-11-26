"""The gap-filling post's notebook and environment file: the rows the notebook's first part
shows, each a number as the post states it, and the versions the environment file pins, those
the run and the figures recorded. The notebook's cells that compute the numbers run in this
test as they are; its first cell, which sets up Colab, and the cells that draw run under nbmake."""
import json
import re
from pathlib import Path

import pytest
from gap_filling_fixtures import scripts_on_path

pytest.importorskip("matplotlib")
scripts_on_path()

ROOT = Path(__file__).resolve().parents[1]
POST = ROOT / "posts" / "2026-11-gap-filling"
NOTEBOOK = POST / "gap_filling.ipynb"

SHOWN = [
    ("days in a row at each place", "730, 730, 730, 730"),
    ("the last post's test tiles entirely filled in", "62%"),
    ("fully-actual pixel-days with a same-day Sentinel-2 or Landsat 8 code", "14%"),
    ("pixel-days fully actual, the four places", "24%"),
    ("pixel-days fully actual, each place", "13%, 21%, 25%, 36%"),
    ("days with every pixel fully actual, of 730", "23, 73, 84, 128"),
    ("days with no pixel fully actual", "81%, 68%, 63%, 48%"),
    ("median gap, for a filled-in pixel-day, days", "19, 11, 8, 6"),
    ("median gap, each gap once, days", "6, 3, 3, 2"),
    ("longest gap, days", "121, 62, 50, 42"),
    ("files checked for embedded fill dates; with any", "8; 0"),
    ("days with maps at each place; images in all", "24, 24, 24, 24; 96"),
    ("pixels fully actual; filled in", "29%; 71%"),
    ("images fully actual in every pixel; entirely filled in; a mix", "14; 55; 27"),
    ("the same two in the last post's records", "14; 55"),
    ("filled pixels with a later recorded source, at least", "51.6%"),
    ("all the pixels filled in with a later source, at least", "36.6%"),
    ("median offset, earlier and later sources, days", "3; 3"),
    ("the farthest, days back and ahead", "63; 93"),
    ("images with some pixels filled in with a later source", "66 of 96"),
    ("images with every pixel filled in with a later source", "13"),
    ("the median image's pixels filled in with a later source", "10%"),
    ("images with none", "30"),
    ("the image", "Yangzhou, 2018-01-01"),
    ("its pixels entirely filled in", "100%"),
    ("its pixels with a later source", "100%"),
    ("their sources, days later", "8 to 43"),
    ("nine in ten within, days", "13"),
    ("images at that share", "13"),
    ("the first day of the series", "2018-01-01"),
    ("map dates on which every window is complete; the first; the last",
     "21; 2018-04-01; 2019-12-01"),
    ("quarters of an image at each place on those dates", "84, 84, 84, 84"),
    ("complete quarters at each place, 15 days; from", "92, 92, 92, 92; 2018-02-01"),
    ("complete quarters at each place, 30 days; from", "92, 92, 92, 92; 2018-02-01"),
    ("complete quarters at each place, 60 days; from", "84, 84, 84, 84; 2018-04-01"),
    ("pixels with a fully-actual day in the 15 days before", "69%, 88%, 86%, 92%"),
    ("pixels with a fully-actual day in the 30 days before", "94%, 96%, 99%, 99.98%"),
    ("pixels with a fully-actual day in the 60 days before", "99.6%, 100%, 100%, 100%"),
    ("the median pixel's fully-actual days in the 30 days before", "4, 6, 7, 11"),
]


def cells():
    nb = json.loads(NOTEBOOK.read_text())
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def first_part():
    """The code cells between the headings of parts 1 and 2."""
    out, inside = [], False
    for kind, text in cells():
        if kind == "markdown" and text.startswith("## "):
            inside = text.startswith("## 1.")
        elif kind == "code" and inside:
            out.append(text)
    return out


def test_the_first_part_shows_the_posts_numbers():
    ns = {"POST": POST, "ROOT": ROOT}
    shown = []
    for text in first_part():
        if "display(" in text:
            continue
        exec(compile(text, str(NOTEBOOK), "exec"), ns)
        if text.rstrip().endswith("table(rows)"):
            shown += ns["rows"]
    assert shown == SHOWN


def test_the_notebook_opens_at_the_release_and_takes_the_frame_off_by_default():
    text = dict.fromkeys(["setup", "frame"], "")
    for kind, source in cells():
        if kind == "code" and "REF = " in source:
            text["setup"] = source
        if kind == "code" and "GET_FRAME = " in source:
            text["frame"] = source
    assert re.search(r'^REF = "2026-11-gap-filling"', text["setup"], re.M)
    assert re.search(r"^GET_FRAME = False$", text["frame"], re.M)
    assert all(c.get("outputs", []) == [] for c in json.loads(NOTEBOOK.read_text())["cells"])


def test_in_colab_the_notebook_uses_only_a_clean_checkout_of_its_release():
    setup = next(source for kind, source in cells() if kind == "code" and "REF = " in source)
    colab = setup.split('if "google.colab" in sys.modules:')[1].split("\nhere = ")[0]
    assert 'Path(f"/content/geoaiwatch-{REF}")' in colab
    assert 'git("rev-parse", f"{REF}^{{commit}}")' in colab and "raise SystemExit" in colab
    assert '"--porcelain", "--untracked-files=no"' in colab
    assert colab.count("check=True") == 2 and "!git" not in colab and "%cd" not in colab
    assert colab.index("raise SystemExit") < colab.index("os.chdir(repo)")


def test_the_environment_file_pins_the_versions_the_run_and_figures_used():
    text = (POST / "environment.yml").read_text().split("dependencies:")[1]
    pins = dict(re.findall(r"^  - ([a-z]+)=([0-9.]+)$", text, re.M))
    run = json.loads((POST / "runs" / "audit.json").read_text())["software"]
    figures = json.loads((POST / "figures" / "figures.json").read_text())["software"]
    assert run["python"].startswith(pins.pop("python") + ".")
    assert pins == {k: run[k] for k in ("numpy", "pandas", "pyarrow", "rasterio")} | \
        {"matplotlib": figures["matplotlib"]}
    assert all(run[k] == figures[k] for k in ("python", "numpy", "pandas", "rasterio"))
