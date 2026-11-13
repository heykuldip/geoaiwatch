# geoaiwatch

[![checks](https://github.com/heykuldip/geoaiwatch/actions/workflows/ci.yml/badge.svg)](https://github.com/heykuldip/geoaiwatch/actions/workflows/ci.yml) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24179266.svg)](https://doi.org/10.5281/zenodo.24179266) [![Latest post](https://img.shields.io/github/v/release/heykuldip/geoaiwatch?label=latest%20post)](https://github.com/heykuldip/geoaiwatch/releases/latest)

Code and results behind the posts on GeoAIWatch, a blog about what satellite imagery and AI can and can't tell us. Each post has its own folder and a tag from the day it went out, so every number in a post can be traced to the code and results it came from.

| Post | Folder |
|---|---|
| A colorful illusion: the invisible color that finds water (November 2026) | [`posts/2026-11-near-infrared`](posts/2026-11-near-infrared) |

## Layout

- `src/geoaiwatch/`: code shared by the posts (the DynamicEarthNet reader, spatial folds, label decoding).
- `posts/<date>-<topic>/`: one post's scripts, design files (`artifacts/`), results (`runs/`), figures and method note (`METHOD.md`).
- `tests/`: what CI runs on every change: a linter, unit tests, tests that recompute every number in each post from its results tables, and each notebook's first part. Retraining, the figures that need imagery or checkpoints, and the dataset download run only by hand, since they need a GPU, the data or a post's release files.

## Install

```
git clone https://github.com/heykuldip/geoaiwatch
cd geoaiwatch
pip install -e ".[test]"              # analysis, results and tests
pip install -e ".[train,figures]"     # training and figures, which need PyTorch
```

Each post's folder has an `environment.yml` with the exact versions its runs used.

Scripts run from the repository root, for example `python posts/2026-11-near-infrared/scripts/analyze.py`.

## Data

The posts use [DynamicEarthNet](https://mediatum.ub.tum.de/1650201) (Toker et al., 2022), published by the Technical University of Munich under CC BY-SA 4.0. Download it from that record, by HTTP, FTP or rsync, and set `DEN_HOME` to the folder that holds `planet/`, or put it at `data/DynamicEarthNet` in the repository. Each post's results tables are enough to reproduce its numbers without the imagery.

## Planet imagery

`src/geoaiwatch/planet.py` searches, orders and downloads PlanetScope scenes through Planet's Python SDK, with your own account: `pip install -e ".[planet]"`, set `PL_API_KEY`, then for example `python -m geoaiwatch.planet search area.geojson 2018-04-01 2018-05-01 --min-clear 90`. The area file holds one area. `order` shows what it would order and places it only with `--yes`; orders are clipped to the area and count against your account's allowance. These are PlanetScope scenes, a different product from the Planet Fusion images in DynamicEarthNet that the posts' experiments use. This repository holds no Planet imagery.

## Licences

| What | Licence |
|---|---|
| Code: `src/`, `posts/*/scripts/`, `tests/` | MIT ([`LICENSE`](LICENSE)) |
| Figures, results tables and checkpoints | CC BY-SA 4.0 ([`LICENSES/CC-BY-SA-4.0.txt`](LICENSES/CC-BY-SA-4.0.txt)), as adaptations of DynamicEarthNet, which carries the same licence |
| Writing: the READMEs and method notes | CC BY-NC-ND 4.0 ([`LICENSES/CC-BY-NC-ND-4.0.txt`](LICENSES/CC-BY-NC-ND-4.0.txt)) |

Figures that show imagery carry the line: Planet Fusion © 2021 Planet Labs Inc. / DynamicEarthNet (Toker et al. 2022) · CC BY-SA 4.0 · modified.

## Citation

Cite the repository with its DOI, [10.5281/zenodo.24179266](https://doi.org/10.5281/zenodo.24179266), which always resolves to the latest version; each post's README gives the DOI of its own release. [`CITATION.cff`](CITATION.cff) has the metadata for citing this repository. For the data, cite Toker et al. (2022), "DynamicEarthNet: Daily Multi-Spectral Satellite Dataset for Semantic Change Segmentation", CVPR 2022 ([arXiv:2203.12560](https://arxiv.org/abs/2203.12560)).
