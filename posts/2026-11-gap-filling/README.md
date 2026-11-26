# Filling in the days: one pixel at a time

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/heykuldip/geoaiwatch/blob/2026-11-gap-filling/posts/2026-11-gap-filling/gap_filling.ipynb) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.TO-COME.svg)](https://doi.org/10.5281/zenodo.TO-COME)

Code and results for the [GeoAI Watch post](https://www.geoaiwatch.com/p/filling-in-the-days) of November 2026. At the four places of the [near-infrared post](../2026-11-near-infrared), it counts how much of DynamicEarthNet's daily Planet Fusion series is marked real, with no gap filling, in the quality file that comes with each day's image. It counts how many pixel-days are fully actual and how long the gaps between them run; on the days with labels, where the one source recorded for a filled pixel lies in time; and how many fully-actual days come before each label date. [`METHOD.md`](METHOD.md) has the full method, what was fixed in advance and in what order, the expectations and how each fared, and where each number in the post comes from.

## What's here

| Path | What it holds |
|---|---|
| `artifacts/plan.json`, `artifacts/plan.md` | the plan, frozen before the run: what to count and how, the checks, the expectations and the example frame's rule |
| `artifacts/amendments.json` | amendment 1, made before any result |
| `scripts/series_plan.py` | reads the plan and its amendment, checking both against their SHA-256 |
| `scripts/series_qa.py` | reads the quality files and classes each pixel-day |
| `scripts/series_run.py` | the run: reads the 2,920 quality files once, checks the plan's invariants, and writes `runs/` |
| `scripts/series_figures.py` | draws the four figures from the tables, checking each number it prints against a second table |
| `scripts/fetch_example.py` | fetches the example frame's two files from the TUM record over FTP, without the whole archive |
| `runs/` | the tables below, `expectations.json` (each expectation, met or not), `audit.json` (the checks, the commit, the versions) and `inputs.sha256` (every dataset file read) |
| `figures/` | the post's four figures, and `figures.json`: the files each render read and wrote, with their SHA-256 |
| `gap_filling.ipynb` | the post's numbers and figures 1, 2 and 4 from the tables; optionally figure 3 from the dataset; it opens in Colab |
| `environment.yml` | the exact library versions the run used |
| `tests/test_gap_filling_*.py`, `tests/gap_filling_fixtures.py` (from the repository root) | the post's tests: the plan and its checks, the run and figures on a small synthetic dataset, every number in the post and `METHOD.md` against the tables, the notebook's rows, and the fetch |

## The tables

All are plain CSV in `runs/`. Columns named `past` and `future` hold what the post calls earlier and later sources. [`TABLES.md`](TABLES.md) says what every column holds, its denominator, and what a blank cell means.

| File | One row per |
|---|---|
| `qa_daily.csv` | place, day and quarter: pixels by Layer 1 class and by Layer 3 status |
| `qa_monthly.csv` | place and month, and place over the two years (period `2018-01..2019-12`): the same counts, with the days in each class |
| `gaps.csv` | place and level: gap lengths per pixel (`pixel`) and runs of days with no fully-actual pixel in the image (`tile_none`) |
| `gap_lengths.csv` | place, level and length: the gaps, censored ones apart. Its `n_days` counts pixel-days at level `pixel` and days at level `tile_none` |
| `label_frames.csv`, `label_cells.csv` | label-date frame, and frame and quarter: fully actual, filled, and filled by Layer 2 source, with the offsets |
| `label_codes.csv` | label-date quarter and Layer 2 code: the filled pixels |
| `label_summary.csv` | the 96 frames pooled, and the example frame |
| `windows.csv` | window, label date and quarter: the pixels covered, and the days in the window with any fully-actual pixel |
| `window_days.csv` | window, label date, quarter and number of fully-actual days: the pixels |
| `block_coverage.csv` | quarter and 90-day window: the pixels covered in 0, 1, 2 and 3 of the 30-day blocks |
| `windows_summary.csv` | place, window and set of label dates (`own`: the window's complete ones; `common`: the 21 on which every window is complete) |

## Start here

Nothing below needs a GPU, PyTorch or a Planet account, and steps 1 and 2 need no download beyond the example frame's two files. From the repository root, in the pinned environment:

```
conda env create -f posts/2026-11-gap-filling/environment.yml
conda activate geoaiwatch-gap-filling
python -m pip install -e ".[test]"
```

The `test` extra adds pytest, the notebook runner and the linter. Or open the notebook in Colab from the badge above, which installs what it needs.

## Reproduce

From the repository root:

1. **The numbers,** from the tables, without imagery: run the notebook's first part, or `pytest tests/test_gap_filling_numbers.py`, which checks every number in the post and `METHOD.md` that comes from the results against the run's files.
2. **The figures:** `python posts/2026-11-gap-filling/scripts/fetch_example.py` fetches the example frame's quality and image files into `DEN_HOME`, then `python posts/2026-11-gap-filling/scripts/series_figures.py --replace` redraws all four. It checks every number it prints against a second table, and the two files against `runs/inputs.sha256`, and stops with nothing written if a check fails. It took one CPU, under a minute and under 1 GB.
3. **Everything,** from the dataset: download the original archive's zones `48N`, `50N` and `52N` from the [TUM record](https://mediatum.ub.tum.de/1650201) (the compressed video release has no quality files); the three zones' archives come to about 66 GB, and the run needs them extracted. Then set `DEN_HOME` to the folder that holds `planet/`, as the root README's *Data* says: for example `export DEN_HOME=/data/DynamicEarthNet`, where the first quality file is `$DEN_HOME/planet/48N/17E-138N/6466_3380_13/PF-QA/2018-01-01.tif`. The run reads the four places' 2,920 PF-QA files and one PF-SR file, the example frame's. `python posts/2026-11-gap-filling/scripts/series_run.py --check-inputs` checks the dates, readability, headers, grids and code ranges of the PF-QA files without counting anything; it doesn't compare the files with those the run read. To check that, byte for byte, run `sha256sum -c` on `runs/inputs.sha256` from `DEN_HOME`: `cd "$DEN_HOME" && sha256sum -c /path/to/geoaiwatch/posts/2026-11-gap-filling/runs/inputs.sha256`. The run reads every file once and writes the tables only if every check passes, and it won't overwrite tables already there, so the committed ones go aside first. It took one CPU, 4 GB of memory and about 8 minutes.
   - **In a git checkout,** the run needs a clean tree and records its commit, so make a branch without the tables: `git switch -c rerun`, `git rm -r -q posts/2026-11-gap-filling/runs` and `git commit -q -m "without the tables"`. Then run `python posts/2026-11-gap-filling/scripts/series_run.py`. `git add posts/2026-11-gap-filling/runs` and `git diff --cached --stat HEAD~1` then list the files that differ from the committed ones; `audit.json` does, since it records the run's own commit and time. To draw the figures from the new tables, commit them first (`git commit -q -m "rerun"`), since step 2 needs them committed and the tree clean; its record then names that commit.
   - **In a copy without git,** such as the release archive, move the tables aside with `mv posts/2026-11-gap-filling/runs runs_committed`, then run `python posts/2026-11-gap-filling/scripts/series_run.py --without-git`, which records the code files' SHA-256 instead of a commit. `diff -rq -x smoke runs_committed posts/2026-11-gap-filling/runs` then lists the tables that differ; `audit.json` does.

The figures, too, expect a git checkout with a clean tree and record its commit; in a copy without git, add `--without-git` to step 2's command, and they record the code files' SHA-256 instead.

`environment.yml` pins the versions the run and the figures used.

To cite this post's code and results: Atwal, K. S. (2026). *geoaiwatch: code and results for the GeoAI Watch posts* (version 2026-11-gap-filling). Zenodo. https://doi.org/10.5281/zenodo.TO-COME

Computing resources were provided by the Office of Research Computing at George Mason University (https://orc.gmu.edu).
