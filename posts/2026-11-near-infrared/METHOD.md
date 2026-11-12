# Method: RGB against RGB plus near-infrared on DynamicEarthNet

Companion to the GeoAIWatch post "A colorful illusion: the invisible color that finds water". It covers the data, the design, what was fixed in advance and when, the evaluation, and where each number in the post comes from.

## Data

[DynamicEarthNet](https://arxiv.org/abs/2203.12560) (Toker et al., 2022) holds daily Planet Fusion surface reflectance in four bands (blue, green, red and near-infrared) at 3 m pixel spacing, with labels on the first of each month. Four places passed the eligibility screen recorded in `artifacts/screen_config.json`: Renshou (`6466`), Fengcheng (`6730`), Yangzhou (`6813`) and Andong (`7026`), with 24 label dates each, or 96 frames. The five classes are agriculture, forest_veg, impervious, soil and water, which the post calls farmland, forest and other vegetation, paved surfaces, bare soil and water.

## Design

Two arms with the same inputs except the fourth channel: RGB plus a channel of zeros, against RGB plus near-infrared, with the zero applied after normalization. Both arms see the same crops, flips and batch order. Normalization statistics (each channel's mean and standard deviation) come from each fold's training pixels only. The model is a U-Net with an ImageNet ResNet-34 encoder (segmentation-models-pytorch 0.3.4). Its first layer starts from ImageNet weights in both arms: the fourth channel copies the red filters, and all four are scaled by 3/4. Training used AdamW (learning rate 1e-4, weight decay 1e-4), batch 16 and eight 256-pixel crops per frame, for 120 epochs. Each training crop lies wholly inside one of the fold's three 480-pixel training squares. The runs were made with the code at the private commit `e0e00a1`, in the environment pinned in `environment.yml`; the training code published here differs from it only in its default output folder, and in checks and records that don't change training. Each frame's eight crops, with their flips and rotations, are drawn once per seed and reused in every epoch; each seed draws its own set. Deterministic algorithms were requested in warn-only mode, and one operation, the forward pass of the CUDA cross-entropy loss, has no deterministic version, so repeated runs agree closely rather than exactly: the first run, when repeated, matched to four decimals. The `deterministic` field in the run files records that request. The run files were written before training recorded its batch counts, so each is tied to its training log by its losses in all 120 epochs, and every log shows 48 batches in each epoch (`runs/grid_isolated/audit.json`, `runs/grid_overlapping/audit.json`).

## What was fixed in advance, and in what order

The commits below are in the private repository where the work was developed. It isn't published because it also holds working notes and paths specific to the computing cluster; this repository holds a cleaned copy of its code and results.

| Commit | What |
|---|---|
| `e08a144` | The design frozen (places, classes, folds and support thresholds), before any model was trained |
| `75e9685` | Two changes before training: the secondary analysis defined on each test cell's gap-fill status, and a fix to how the QA band holding each pixel's distance in days to its Sentinel-2 or Landsat-8 reference is decoded |
| `4335ce9` | Evaluation at epochs 20, 40, 80 and 120, with 120 primary, set before any held-out evaluation |
| `0dafa50` | The ±0.02 threshold, the agriculture↔forest confusion diagnostic and the expectation below written into `scripts/train.py`, after the first of the twelve runs (fold NW, seed 0) had been evaluated; the grid was then cancelled and rerun from scratch, and that first run reproduced to four decimals |
| `9309a72` | The grid's result, computed then by pooling confusion matrices |
| `7f9314a` | The cell-level estimand made primary, when both it and the pooled estimate were known |
| `e0e00a1` | Training crops confined to the training squares, recorded before any rerun; the twelve runs repeated, with results in `7a20e51` |

A pre-registration scaffold in the private repository holds the same threshold and diagnostic, but it entered that history only after the first results, so the record doesn't date its contents earlier. The expectation, as written in `scripts/train.py`:

> "NIR should help vegetation vs non-vegetated but not necessarily agriculture vs forest, since both are vegetated and a single date carries no phenology."

Compared with that expectation, forest_veg gained +0.047 and agriculture +0.040, both at or above +0.02 in all four folds. Agriculture→forest confusion fell by −0.017, while forest→agriculture rose by +0.003. These class-level gains don't directly test the expected separation of vegetation from non-vegetated surfaces, and the expectation didn't predict a null result for agriculture against forest. Water, the largest gain at +0.382, isn't mentioned. By fold, the classes at or above +0.02 were water, forest_veg, soil and agriculture in all 4, and impervious in 1.

## Evaluation

Spatial folds: each frame's four quadrants are held out in turn, giving 96 test cells (place × date × quadrant) per fold and 384 in all, with three seeds per fold. Training quadrants stop 32 pixels short of the held-out quadrant's edges, and training crops lie wholly inside them. The hold-out is spatial, within the same four places, so it says nothing about new places; and the folds share most of their training pixels, so neither folds nor seeds are independent replicates. The primary estimand averages the seeds within each cell and weights cells equally within a fold: ΔmIoU +0.101, positive in all four folds (+0.091 to +0.115; fold sd 0.010; within-fold seed sd 0.003–0.017), and positive in 382 of 384 cells. Pooling the confusion matrices instead gives +0.110. The cell-level estimand follows the scaffold, and became primary when both values were known. The held-out gain was stable across the checkpoints set before evaluation, +0.100, +0.099, +0.097 and +0.101 at epochs 20, 40, 80 and 120, which shows robustness to training length rather than generalization. Every test cell holds at least 1,024 pixels of every class (the smallest count, 1,674, is water). For context only: DynamicEarthNet's three temporal baselines report validation mIoU between 29.7 and 35.7 over six evaluated classes (snow and ice excluded), using whole image sequences (Toker et al., 2022, Table 3); the setups differ too much for the levels here to be compared with them.

The runs reported here are a repeat. Later testing found that the earlier runs' training crops were drawn anywhere in the frame, so about a quarter of their pixels, unlabeled, came from the held-out quadrant. The crops were then confined to the training squares and all twelve runs repeated, with everything else unchanged. The earlier runs gave ΔmIoU +0.063 (mIoU 0.310 → 0.373); they and their tables are in `runs/grid_overlapping/` and `runs/overlapping/`.

## Gap filling

Planet Fusion fills gaps, from clouds or from days without an acquisition, using observations from other dates. Of the 384 test cells, 237 (62%) are fully gap-filled. Mean ΔmIoU was +0.098 on the 70 cells with no gap filling and +0.100 on the 237 fully gap-filled ones (the correlation between a cell's gain and its mean gap-fill score is r = −0.030 over all 384 cells), but gap-fill status covaries with place, date and season.

## The example frame

Figures 1–3 show Yangzhou, 2018-04-01, NW quadrant, chosen after the water gains by place had been seen. Its panels come from seed 0 (mIoU 0.278 → 0.415). Over three seeds its gain is +0.136; 82% of test cells gain less, and it is 1.3 times the median (+0.105). The frame was kept when the runs were repeated, although it now stands out less. In Figure 5, five of Andong's six lowest dates fall between December and March and the sixth in November (its January and February gains average +0.261 and +0.267, against +0.648 in July); that was noticed after the results and is provisional.


## Where the numbers come from

| Number or statement | Where |
|---|---|
| Places, dates, classes and folds | `artifacts/screen_config.json`, `artifacts/screen_result.json` |
| mIoU 0.337 → 0.438 (+0.101); positive in all four folds and in 382 of 384 cells; the fold range and standard deviations | `runs/analysis.txt`, primary result, and the epoch-120 row of checkpoint sensitivity for the two means |
| Gains by class (+0.382, +0.047, +0.040, +0.024, +0.009), the folds at or above +0.02, and water's share of the gain | `runs/analysis.txt`, primary result, "across folds, per class" |
| The pooled estimate, +0.110 | `runs/analysis.txt`, sensitivity |
| Gains at epochs 20, 40, 80 and 120 | `runs/analysis.txt`, checkpoint sensitivity |
| Agriculture→forest −0.017 and forest→agriculture +0.003 | `runs/analysis.txt`, diagnostics |
| 1,024 and 1,674 pixels | `runs/analysis.txt`, opening lines |
| 237 of 384 cells fully gap-filled; +0.098 and +0.100; r = −0.030 | `runs/analysis.txt`, QA-status heterogeneity; `artifacts/cell_qa_status.parquet` |
| The example frame's seed-0 mIoU 0.278 → 0.415 and water IoU 0.129 → 0.601; the 15% and 7% in Figure 3 | `scripts/figures.py`, which computes them from the released checkpoints and checks both arms' mIoU against `runs/cell_level.csv` and the two shares against `runs/figure3_shares.json` |
| Its three-seed gain +0.136, with 82% of cells gaining less, and 1.3 times the median (+0.105) | `runs/frame_stats.txt` |
| Water gains by place (+0.194 Renshou, +0.399 Fengcheng, +0.432 Yangzhou, +0.504 Andong) and by date | `runs/cell_level.csv`, as plotted in Figure 5 |
| Training settings | `scripts/train.py`, and the settings stored in each `runs/grid_isolated/*.json` |
| The earlier runs' values | `runs/overlapping/analysis.txt` and `runs/overlapping/frame_stats.txt`, from the runs in `runs/grid_overlapping/` |
| The earlier runs' share of held-out pixels in training crops | Replaying the earlier sampler's seeded positions, uniform over the frame, against the masks in `src/geoaiwatch/folds.py` |
| The baselines' validation mIoU, 29.7 to 35.7 | Toker et al. (2022), Table 3 |

## Licences

Text: CC BY-NC-ND 4.0. Figures and results tables: CC BY-SA 4.0. Code: MIT. Imagery in the figures: Planet Fusion © 2021 Planet Labs Inc. / DynamicEarthNet (Toker et al. 2022) · CC BY-SA 4.0 · modified.
