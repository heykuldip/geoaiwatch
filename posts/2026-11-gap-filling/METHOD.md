# Method: how much of DynamicEarthNet's daily series is marked real, at four places

Companion to the GeoAI Watch post "Filling in the days: one pixel at a time". It covers the data, what was counted and how, what was fixed in advance and in what order, the expectations and how each fared, the limits, and where each number in the post comes from.

## Data

[DynamicEarthNet](https://arxiv.org/abs/2203.12560) (Toker et al., 2022) gives each of its places a Planet Fusion surface-reflectance image (PF-SR) and a quality file (PF-QA) for every day of 2018 and 2019. The quality files come with the dataset's original archive at [TUM](https://mediatum.ub.tum.de/1650201), not with its compressed video release. This analysis reads them for the four places of the [near-infrared post](../2026-11-near-infrared): Renshou (`6466`), Fengcheng (`6730`), Yangzhou (`6813`) and Andong (`7026`). That's one file a day from 2018-01-01 to 2019-12-31, 730 a place and 2,920 in all, each 1,024 × 1,024 pixels at 3 m pixel spacing and split into the near-infrared post's four 512 × 512 quarters. Each quality file has nine layers, and this analysis reads the first three. The labels are for the first day of each month, so there are 24 label dates a place and 96 label-date frames. PF-SR is read only for the example frame: the run checks its file, and the figures' render draws it. The run stops if a day's file is missing, duplicated or unreadable, and fills or interpolates nothing; `runs/inputs.sha256` lists every file it read, with its SHA-256.

The codes are read as Planet's [*Planet Fusion Monitoring Technical Specification*](https://web.archive.org/web/20220111190640/https://assets.planet.com/docs/Planet_fusion_specification_March_2021.pdf), version 1.0.0-beta.3 of March 2021, defines them (Table 5, §3.2 and §4.4). Planet's own address for it no longer answers; the link is to the Internet Archive's copy of 11 January 2022, the same file byte for byte as the one read for this analysis. The files don't say which version they follow. The specification says each quality file embeds the dates of every scene used to fill it, but the first and last files of each place, 8 of the 2,920, hold no metadata beyond their image structure (`runs/audit.json`). The run's range checks reject codes outside the documented ranges, and its label-date checks reproduce the near-infrared post's QA counts; neither establishes what the codes mean, or which version of the specification the files follow.

## What the quality layers say

L1, L2 and L3 are layers 1, 2 and 3 of one pixel on one day, a pixel-day. Every test is on the files' integer codes, with no tolerance, and each pixel-day is exactly one of nodata, fully actual or filled, tested in that order.

| Term | Test |
|---|---|
| nodata | L1, L2 or L3 is −1000 |
| fully actual | not nodata, and L1 = 1: in the specification's words, "no gap-filling (i.e., real PS/S2/L8 data)" |
| filled | not nodata, and L1 from 2 to 100: entirely filled at 100, blended from 2 to 99 |
| Layer 2 source, same day | L2 is 0, 900 (Sentinel-2 data of the day used to fill) or −900 (Landsat 8 data of the day used to fill) |
| Layer 2 source, earlier | L2 from −899 to −1, an offset of L2 days |
| Layer 2 source, later | L2 from 1 to 899 |
| Layer 3 status | −999 no PlanetScope data; 2 to 6 cloud, shadow or haze; 7 band misalignment; 1 marked clear |
| day class | all: every pixel of the image fully actual; none: no pixel fully actual; some: neither |
| gap | for one pixel, a run of consecutive days that aren't fully actual. A gap that includes 2018-01-01 or 2019-12-31 is censored: counted, and left out of the summaries, as are runs of days in class none that touch either end |
| window | a set of days before a label date; complete when every day lies in 2018–2019 |
| covered | a pixel with at least one fully-actual day in the window |

The tables call earlier and later sources `past` and `future`.

For a filled pixel, Layer 2 gives one source, and the specification describes it twice. Table 5 calls it the scene with actual data at the shortest distance in days. §4.4 says the field identifies only the tile with the highest weight in the fill, and that tiles from other dates may also feed in, typically up to six from before and after the date, with their weights falling as their distance from it grows; anomalous pixels may also be smoothed with the days either side. The two descriptions aren't reconciled, and this analysis doesn't assume they name the same tile. So the share of filled pixels whose source is later is a quality indicator, not the share of a fill's information that comes from later days. If the reported later tile contributes to that pixel's estimate, as §4.4's reading implies, that share is a lower bound on the share of filled pixels with any later-dated input: at the label dates, at least 51.6% of filled pixels had one, or 36.6% of all label-date pixels, as the post gives it. Table 5's wording alone doesn't establish that.

On fully-actual pixel-days, the specification's Figure 4 gives Layer 2 as 0, but Table 5 and §3.2 let Sentinel-2 or Landsat 8 data of the day fill PlanetScope's gaps, coded 900 or −900, and Layer 1 counts those data as actual. Of the 733,982,479 fully-actual pixel-days, Layer 2 is 0 on 86%, 900 on 11% and −900 on 3%. At each place, Layer 3 is 1 (clear) on exactly as many fully-actual pixel-days as carry the code 0, and otherwise −999 or 2 to 6, not 7 (`runs/audit.json`, diagnostics). So about 14% of fully-actual pixel-days carry a same-day Sentinel-2 or Landsat 8 code, and as many have Layer 3 marking PlanetScope data missing or masked; the tables don't record whether these are the same pixel-days.

Layer 3 is Planet's PlanetScope cloud and shadow mask (Table 5), so its status beside a filled pixel describes the PlanetScope data, not why Fusion filled the pixel.

## What was counted

1. **Real share:** the share of pixel-days fully actual, by place and month and over the two years, and the days in each day class.
2. **Gaps:** by place, gap lengths weighted by pixel-days (the gap each not-fully-actual pixel-day lies in) and by gaps (each gap once): the median, 90th percentile and longest, with censored gaps counted apart; and the runs of days in class none.
3. **Label dates:** for each of the 96 frames and their 384 quarters, the shares fully actual and filled, the shares of filled pixels by Layer 2 source, and the offsets of earlier and later sources.
4. **Windows:** for each quarter and label date, windows of 15, 30, 60 and 90 days ending the day before the label date, and three 30-day blocks, days 1–30, 31–60 and 61–90 before it. For each, the covered share of the quarter's 262,144 pixels and the pixels' numbers of fully-actual days; for the 90 days, the pixels covered in 0, 1, 2 or all 3 blocks. A window counts only where it's complete: on 23 label dates a place (from 2018-02-01) for 15 and 30 days and the first block, and on 21 (from 2018-04-01) for the others; none is complete for 2018-01-01. Comparisons across windows, as in figure 4, use the 21 label dates on which every window is complete.
5. **Layer 3 status of filled pixel-days,** entirely filled and blended apart, by place and month.

**Weights.** Pixel-weighted values come first: a share pools the pixel-days or pixels it summarizes. Date-weighted values, the share within each day, frame or quarter averaged with equal weights, are reported beside them wherever the two round to different whole percentages. Every day has 1,048,576 pixels and every quarter 262,144, so the fully-actual share of pixel-days and the covered share of a window are the same under both weights; every label-date frame has 1,048,576 pixels too, so the label dates' shares fully actual and filled are as well. A share of filled pixels, such as a Layer 2 source's or a Layer 3 status's, is date-weighted over the days or frames that have at least one pixel of the kind it divides by: for the label dates' Layer 2 sources, the 82 frames with any filled pixel. Offset quantiles weight filled pixels, earlier and later sources apart. A quantile is the smallest observed value whose cumulative weight reaches the given fraction (the inverted CDF), so each is an observed whole number of days, and at least nine in ten values lie at or below a 90th percentile.

## What was fixed in advance, and in what order

The commits below are in the private repository where the work was developed. It isn't published because it also holds working notes and paths specific to the computing cluster; this repository holds a cleaned copy of its code and results. Their order is the author's record and can't be checked from here. What can be checked here is the files: `runs/audit.json` and `figures/figures.json` give the SHA-256 of every file they used, and each matches the file in this folder, with the exception the paragraph on the folder's old name explains (*The run and the figures*).

| Commit | What |
|---|---|
| `27d095e` | The plan frozen, `artifacts/plan.json` and `plan.md`: what to count and how, the checks, the expectations below and the example frame's rule, before the daily series of these four places was analyzed |
| `f182667` | Amendment 1, `artifacts/amendments.json`, before any result: three tables added (`label_summary.csv`, `label_codes.csv`, `window_days.csv`) and nine readings of the plan fixed |
| `40a8b42` | The code the run used |
| `ba3315f` | The run's results, `runs/` |
| `2a6bea4` | The figures' code, written after the results |
| `e913b9d` | The figures, drawn from the results |
| `e53e891`, `c5aaa00` | The colorbar labels turned and figure 4's footer set in shorter lines, and the figures redrawn |
| `7021394`, `59fd492` | The figures' code tidied without changing what it draws, and the figures redrawn from it, byte for byte as at `c5aaa00` |
| `d7b44ee`, `40401b7` | The figures' text put in the post's words, and the figures redrawn from it; what they show is unchanged |
| `75d4690`, `0ad7c24` | The figures' code checking the rest of their printed numbers against a second table, and figure 2's footer naming the nodata code; the figures redrawn from it, figures 1, 3 and 4 byte for byte as at `40401b7` |
| `8d9a8e5`, `56bd385` | The figures' text in the post's words again (map dates, images, the orange line), figure 3's legend limited to what the image holds, and figure 2's nodata sentence kept for when a pixel carries the code; the figures redrawn from it, the plots of figures 1, 2 and 4 pixel for pixel as before, their footers changed |

## The expectations, and how each fared

The plan wrote down seven expectations before the run, with the exposure listed under *Already seen*, so they're expectations informed by exploratory work, not blind predictions; the second to the fifth draw on the label dates, which are part of the series counted. Each was judged on exact counts, before rounding (`runs/expectations.json`). None was stated for the label dates' Layer 2 sources, since summaries of them had been computed before the plan.

| | Expectation, as `plan.json` words it | Observed | |
|---|---|---|---|
| EX4 | "At every place, fewer than a third of days are in class some." | Andong had 253 of 730 days in class some (35%); Renshou 117, Fengcheng 160, Yangzhou 183 | not met |
| EX1 | "No pixel-day in the 2,920 files is nodata in layers 1-3." | none was | met |
| EX2 | "Pooled over the four places, the share of pixel-days fully actual is at least 15% and at most 40%." | 24% | met |
| EX3 | "Andong's share of pixel-days fully actual is higher than at each of the other three places." | Andong 36%, against 25%, 21% and 13% | met |
| EX5 | "Gaps are longest at Renshou: its median gap length, weighted by pixel-days, is larger than at each of the other three places." | 19 days, against 11, 8 and 6 | met |
| EX6 | "Among filled pixel-days, the share with no PlanetScope data (L3 = -999) is higher at Renshou than at any other place, and lower at Andong than at any other place." | 91% at Renshou and 72% at Andong, against 84% and 82% | met |
| EX7 | "For the 30-day nested window, the median covered share over a place's complete cells is at least 80% at every place." | 100% at every place, over 92 quarters each | met |

## Already seen

Before the plan was frozen, as it lists them:

- Layer 1 classes of the 96 label-date frames and their 384 quarters, by exact codes (the near-infrared post's `artifacts/cell_qa_status.parquet` and `frame_endpoints.parquet`), and the frame counts in its `screen_config.json`.
- Layer 2 at the 96 label dates: per-frame median and 90th-percentile offsets (its `frame_qa_summary.parquet`), and the counts of same-day codes.
- In unpublished work: eight label-date frames in detail, by Layers 1–3; the quality files on at least seven other days, the acquisition dates of PlanetScope scenes examined there; and PlanetScope's acquisition frequency over the four places in Planet's catalog. Which frames and days these were comes from Planet's catalog, which isn't published here, so they aren't listed.
- An exploratory analysis of the daily series at four other places, in the Americas: PlanetScope coverage close to all or nothing on each day, and cloudier weather on days without PlanetScope data.
- Not seen: the daily series of these four places, apart from the days above.

## More numbers

The post keeps to a few numbers. These are the rest, from the same tables; the four places are in the order Renshou, Fengcheng, Yangzhou, Andong throughout.

- **Days** with every pixel fully actual, some, or none: 23, 117 and 590 of 730; 73, 160 and 497; 84, 183 and 463; 128, 253 and 349 (`qa_monthly.csv`).
- **Months.** Every place has fully-actual pixel-days in every month. The lowest month: 0.8% (2019-11), 2% (2019-02), 8% (2019-02) and 9% (2019-07); the highest: 32% (2018-04), 61% (2019-09), 47% (2018-10) and 65% (2018-02) (`qa_monthly.csv`).
- **Gaps,** in days, as median, 90th percentile and longest. Weighted by pixel-days: 19, 37, 121; 11, 42, 62; 8, 27, 50; 6, 21, 42. Each gap once: 6, 23, 121; 3, 15, 62; 3, 12, 50; 2, 7, 42. Censored, and left out: at the start 1,042,157, 1,048,576, 1,048,576 and 77,253 gaps; at the end 1,048,576, 1,048,576, 2,348 and 1,048,576 (`gaps.csv`, level `pixel`).
- **Whole-image runs** of days with no fully-actual pixel: median 4, 2, 2 and 2 days; 90th percentile 17, 10, 8 and 5; longest 39, 34, 21 and 10. Censored, and left out: 1 at Renshou (at the end), 2 at Fengcheng (start and end), 1 at Yangzhou (start), 1 at Andong (end) (`gaps.csv`, level `tile_none`).
- **Label dates.** Filled pixels by Layer 2 source, pooled: earlier 45%, the same day 3% and later 51.6%; weighting each of the 82 frames with filled pixels equally, 41%, 11% and 48%. Same-day sources by code, of filled pixels: 0 on 2%, 900 on 0.8% and −900 on 0.2%; weighting each frame equally, 9%, 2% and 0.2%. Earlier sources: median 3 days, 90th percentile 7, farthest 63 (Renshou, 2018-07-01, SW quarter, 7,260 pixels). Later sources: median 3, 90th percentile 11, farthest 93 (Renshou, 2018-08-01, SW quarter, 5,310 pixels). Of the 82 frames with filled pixels, 66 have some with a later source and 60 some with an earlier one; the later share of filled pixels is over half in 36 frames and the earlier in 37; in 10 frames every filled pixel has an earlier source (`label_summary.csv`, `label_codes.csv`, `label_frames.csv`).
- **Windows,** pooled over each window's own complete quarters (`windows_summary.csv`, dates `own`). Covered: 15 days, 69%, 88%, 86% and 92%; 30 days, 94%, 96%, 99% and 99.98% (5,070 of Andong's 24,117,248 pixels not covered); 60 days, 99.6% (82,185 of Renshou's pixels not covered) and 100% at the others; 90 days, 99.9% (14,740 not covered) and 100%. Quarters with no pixel covered: in 15 days, 22, 5, 4 and 0; in 30 days, one, at Fengcheng (2019-03-01, NW). Fully-actual days per pixel, 10th, 50th and 90th percentiles: in 15 days 0/1/5, 0/3/6, 0/4/6 and 1/5/10; in 30 days 1/4/8, 2/6/12, 4/7/11 and 4/11/17; in 60 days 4/7/13, 5/11/23, 10/14/20 and 9/21/31; in 90 days 6/13/18, 8/18/36, 16/22/29 and 17/33/45.
- **The three 30-day blocks** (84 quarters a place; `block_coverage.csv`). Pixels with a fully-actual day in all three: 90%, 88%, 99% and 99.95%; in one or none, only at Renshou (162,777 and 14,740 pixels). Quarters in which some pixel misses a block: 34, 24, 5 and 4 of 84; the lowest share covered in all three: 2% (Renshou, 2019-12-01, SW), 0% (Fengcheng, 2019-03-01, NW), 55% (Yangzhou, 2018-04-01, SW) and 99% (Andong, 2019-11-01, SW).
- **Layer 3 on filled pixel-days** (`qa_monthly.csv`; `qa_daily.csv` for the pooled date-weighted values). Entirely filled and blended together, with the date-weighted values in parentheses, over the 707, 657, 646 and 602 days with filled pixels: no PlanetScope data on 91%, 84%, 82% and 72% (82%, 75%, 71% and 57%); cloud, shadow or haze on 7%, 12%, 14% and 21% (9%, 14%, 16% and 23%); marked clear on 3%, 4%, 5% and 7% (9%, 11%, 13% and 20%). Entirely filled: no PlanetScope data on 93%, 87%, 85% and 77% (date-weighted 82%, 77%, 73% and 59%). Blended, 2–7% of filled pixel-days: marked clear on 79%, 63%, 70% and 76% (date-weighted 81%, 66%, 73% and 78%). Misaligned: under 0.02% everywhere (date-weighted, under 0.03%).

## The example frame

The plan's rule: "The label-date frame with the largest share of its pixels filled and with a future Layer 2 source; a tie goes to the earliest label date, then to the lowest cube number." Thirteen of the 96 frames share the maximum, 100%, and the tie-break picks Yangzhou, 2018-01-01, the series' first day. Every one of its 1,048,576 pixels is entirely filled, with a later source 8 to 43 days ahead (median 10, 90th percentile 13). On that date, the earlier sources at the other three places all lie one day back, on 2017-12-31, against up to 63 days back on later dates, so the start of the series may limit earlier sources there; that's a possibility, not something tested. Summaries of the label dates' Layer 2 had been computed before the plan, so the frame the rule picks may have been foreseeable; the rule picked it, not a look. Across the 96 frames, the median share is 10%, and 30 frames have none. The picture draws PF-SR bands 3, 2 and 1 (red, green and blue on disk), each stretched between its 1st and 99th percentiles, with gamma 0.75, as the near-infrared post drew its frame.

## Limits

- **Fully actual isn't a clean cut at the day.** Fusion calibrates its data against Sentinel-2 and Landsat 8 references sampled from a window typically a year wide, centered on the period processed (§4.3); it aligns images to a reference made from four mostly clear images around the date, which can be later ones (§4.5); it may smooth anomalous pixels with the days either side (§4.4); and this series was made with later days at hand, as its later Layer 2 sources show, which §4.7 describes as processing in backfill. Inputs restricted to fully-actual pixels on days up to the label date are called reduced-look-ahead: the restriction leaves out filled pixels, but it doesn't rule out later information, and nothing here measures how much of it remains.
- **One recorded source per filled pixel,** described two ways that the specification doesn't reconcile (above).
- **The codes' version** isn't recorded in the files (above).
- **Four places,** chosen for the near-infrared post by a screen for label support (its `artifacts/screen_config.json`), don't stand for DynamicEarthNet as a whole.
- **No model.** The post counts; it doesn't test what filled pixels do to a model.

## The run and the figures

`scripts/series_run.py` read the 2,920 quality files once, and the example frame's PF-SR file, passed the plan's 13 checks and four more, and wrote `runs/`, in about 8 minutes on one CPU with 4 GB of memory, with Python 3.11.16, numpy 1.26.4, pandas 3.0.5 and rasterio 1.4.4 with GDAL 3.12.3. `runs/audit.json` records the checks, the commit, the SHA-256 of `plan.json` and `amendments.json`, and the versions. `scripts/series_figures.py` drew the figures from the tables in under a minute on one CPU, with under 1 GB, checking every number it prints against a second table and the example frame's files against `runs/inputs.sha256`; `figures/figures.json` records the files it read and wrote, with their SHA-256.

Until the release, the post's folder was named `2026-12-gap-filling`. `artifacts/amendments.json` and `runs/audit.json` keep the paths they recorded under that name. The usage line at the top of `scripts/series_run.py` was then changed to the new name, so that file differs from the SHA-256 `runs/audit.json` gives it in that line only: with the old name back in that line, it matches. `figures/figures.json` comes from a later render, under the new name, and its SHA-256 match the files as they are.

## Where the numbers come from

| Number or statement in the post | Where |
|---|---|
| 730 days in a row | `runs/audit.json`: `inputs.pf_qa_files`, 730 a place, and `checks.I1`, the check of one file for each day |
| 24%; 13%, 21%, 25% and 36% | `runs/qa_monthly.csv`, period `2018-01..2019-12`: `n_fully_actual` over `n_pixel_days` |
| 23, 73, 84 and 128 of 730 days; 81%, 68%, 63% and 48% of days | `qa_monthly.csv`, period `2018-01..2019-12`: `days_all`, `days_none` |
| 14% of fully-actual pixel-days with a same-day Sentinel-2 or Landsat 8 code | `runs/audit.json`, diagnostics |
| 19, 11, 8 and 6 days; 6, 3, 3 and 2; 121, 62, 50 and 42 | `runs/gaps.csv`, level `pixel` |
| 29% and 71%; 51.6% of filled pixels with a later source; 36.6% of all pixels | `runs/label_summary.csv` |
| 14, 55 and 27 frames; 66 and 30 of the 96; 13; 10% | `runs/label_frames.csv`; `label_summary.csv` |
| Matching the records from the near-infrared post | `runs/audit.json`: `checks.I9`, which compares the label dates' Layer 1 classes with that post's committed tables, its `artifacts/cell_qa_status.parquet` row by row |
| a median of 3 days either way; 63 and 93 days | `label_summary.csv`; `runs/label_codes.csv` |
| Yangzhou, 2018-01-01; 8 to 43 days, at least nine in ten within 13; 13 frames at 100% | `label_summary.csv`; `label_frames.csv`; `label_codes.csv` |
| 69%; 86% to 92%; 94%; 96% or more; 99.6%; every pixel at the other three; 92 and 84 quarters a place | `runs/windows_summary.csv`, dates `own` |
| Figure 4's 21 map dates and 84 quarters a place | `windows_summary.csv`, dates `common` |
| 4 to 11 fully-actual days in 30 | `windows_summary.csv`; `runs/window_days.csv` |
| Figure 4's counts under each window | `runs/windows.csv`: complete quarters on the 21 common label dates whose `n_covered` equals `n_pixels` |
| The eight files checked, each place's first and last, without embedded dates | `runs/audit.json`: `inputs.metadata` |
| 62% of the near-infrared post's test tiles | that post's `artifacts/cell_qa_status.parquet` |
| What the specification says | [*Planet Fusion Monitoring Technical Specification*](https://web.archive.org/web/20220111190640/https://assets.planet.com/docs/Planet_fusion_specification_March_2021.pdf), version 1.0.0-beta.3, March 2021: Table 5, §3.2, §4.3, §4.4, §4.5 and §4.7 |

## Licences

Text: CC BY-NC-ND 4.0. Figures and results tables: CC BY-SA 4.0. Code: MIT. Imagery in the figures: Planet Fusion © 2021 Planet Labs Inc. / DynamicEarthNet (Toker et al. 2022) · CC BY-SA 4.0 · modified.
