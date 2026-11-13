# A colorful illusion: the invisible color that finds water

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/heykuldip/geoaiwatch/blob/2026-11-near-infrared.1/posts/2026-11-near-infrared/notebook.ipynb)

Code and results for the GeoAIWatch post of November 2026. Two identical models learned to map land cover from DynamicEarthNet imagery, one from red, green and blue only and one with near-infrared added. Near-infrared helped most with water. [`METHOD.md`](METHOD.md) has the full method, what was fixed in advance and when, and where each number in the post comes from.

## What's here

| Path | What it holds |
|---|---|
| `scripts/train.py` | trains both models for one fold and seed |
| `scripts/analyze.py` | reads the twelve runs and prints the results (`runs/analysis.txt`) |
| `scripts/figures.py` | draws the figures |
| `scripts/frame_stats.py` | statistics for the figure frames (`runs/frame_stats.txt`) |
| `scripts/verify_setup.py` | checks that a full local copy of the dataset matches the design files, before retraining |
| `artifacts/` | the fixed design (`screen_config.json`, `screen_result.json`) and the data checks; [`artifacts/README.md`](artifacts/README.md) explains them |
| `runs/grid_isolated/` | the twelve runs behind the post, one file per fold and seed: settings and metrics at each evaluated epoch; `audit.json` ties each run to its training log |
| `runs/grid_overlapping/`, `runs/overlapping/` | an earlier set of runs whose training crops overlapped the held-out quarter, with its tables, kept for comparison (`METHOD.md`) |
| `runs/cell_level.csv` | metrics for each of the 384 test cells, the table the post's numbers come from |
| `runs/figure3_shares.json` | Figure 3's two shares, which `scripts/figures.py` checks its render against |
| `figures/` | the post's figures, plus a detailed Figure 4 by test quarter |
| `notebook.ipynb` | the post's numbers and figures 4 and 5 from the tables; optionally the dataset and figures 1 to 3, or your own Planet scenes; it opens in Colab |
| `scripts/fetch_frame.py` | reads the example frame and its labels from the TUM record over FTP, without the whole archives |
| `environment.yml` | the exact library versions the runs used |

## Reproduce

From the repository root:

1. **The numbers**, from the twelve run files in `runs/grid_isolated/`, without imagery: `python posts/2026-11-near-infrared/scripts/analyze.py`. `tests/test_near_infrared_numbers.py` checks every number in the post against the results tables.
2. **The figures**, with the dataset (`DEN_HOME`) and the two checkpoints attached to [this post's release](https://github.com/heykuldip/geoaiwatch/releases/tag/2026-11-near-infrared.1), placed in `runs/models/`: `python posts/2026-11-near-infrared/scripts/figures.py --fig all --tolerance 1e-3`. Their SHA-256 checksums are `eade4cc1a6e56c46cd5acfa67f7fa7b32560e4cea522a14ecdec3a96316cabb9` for `NW_s0_rgbn_ep120.pt` and `8447a35f46eb6bbf14fec2917b3e33abd95a7de0374120362096219b48a35ee4` for `NW_s0_rgb0_ep120.pt`. `figures.py` stops if a checkpoint is missing or holds another fold, seed or epoch, and checks the rendered panels against `runs/cell_level.csv` and `runs/figure3_shares.json`. `scripts/fetch_frame.py` fetches just the frame these figures need. Small differences are expected on other hardware. On another GPU or a CPU, the panels' overlap score can differ from the committed one in the fifth decimal, which the tolerance allows, and a few hundred to a few thousand pixels of Figures 2 and 3 can fall in another class: 466 and 2,227 on an NVIDIA A100, against the committed H100 render. Processor rounding in the brightness curve can change a few pixels of the true-color images (Figure 1a and the combined Figure 1) by one intensity level.
3. **Everything**, by retraining the twelve runs on a GPU, with the full dataset: check it first with `python posts/2026-11-near-infrared/scripts/verify_setup.py`, then run `python posts/2026-11-near-infrared/scripts/train.py --fold NW --seed 0`, and so on for the folds NE, NW, SE and SW and the seeds 0, 1 and 2, adding `--save-models` for fold NW, seed 0, whose checkpoints the figures use. The defaults are the post's settings, and each run rewrites its file in `runs/grid_isolated/`, so `git diff` shows any difference from the committed runs. Then steps 1 and 2.

`environment.yml` pins the versions the runs used: `conda env create -f posts/2026-11-near-infrared/environment.yml`, then `conda activate geoaiwatch-near-infrared`, then `python -m pip install -e .` from the repository root. The model's encoder starts from torchvision's ImageNet ResNet-34 weights, `resnet34-333f7ec4.pth`, which segmentation-models-pytorch downloads on first use; the file name carries the first eight characters of its SHA-256.

Retraining needs a CUDA GPU. Each run keeps the 96 frames in memory, 1.69 GiB, and used at most 2.29 GiB of GPU memory (batch 16, 256-pixel crops); the runs here were made on NVIDIA H100s. `--smoke` trains for two epochs on eight frames, as a quick check of a setup.

To cite this post's code and results, use the DOI of its latest release, listed on Zenodo under the repository's DOI, [10.5281/zenodo.24179266](https://doi.org/10.5281/zenodo.24179266).

Computing resources were provided by the Office of Research Computing at George Mason University (https://orc.gmu.edu).
