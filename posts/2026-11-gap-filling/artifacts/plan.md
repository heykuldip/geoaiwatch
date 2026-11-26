# Post 2 plan: how much of the daily series is real

This plan fixes what Post 2 measures and how, before the analysis of the complete daily series at these four places is run; some of the same data had been examined before, as *Already seen* sets out. The code reads `plan.json`, which holds the same values; where the two differ, `plan.json` governs and the difference is an error to fix. A change after the plan is frozen is a dated amendment in `artifacts/amendments.json`, with its reason and whether any result had been seen.

## Question

DynamicEarthNet gives each place a Planet Fusion image for every day of 2018 and 2019. At Post 1's four places, how many of those days hold real observations, how long are the gaps between them, and when a pixel is filled in on a label date, is the one source its QA layer records the same day, in the past or in the future?

The answers are descriptive and stop at these four places: no model, no estimate for DynamicEarthNet as a whole, and no data from outside the dataset.

## Data

PF-QA, Fusion's quality layers (Planet Fusion specification, March 2021, Table 5), for `6466` Renshou, `6730` Fengcheng, `6813` Yangzhou and `7026` Andong: one 1024 × 1024 file a day from 2018-01-01 to 2019-12-31, 730 a place. Only layers 1–3 are read, and PF-SR only for the example frame's picture. The label dates are the first of each month, 24 a place and 96 in all. Each tile splits into Post 1's four 512 × 512 quarters (NW, NE, SW, SE).

The run needs exactly one readable file per place and day. A missing, duplicated or unreadable file stops the run and is listed; no day is filled or interpolated. The SHA-256 of every file read is recorded, and so are the metadata tags of each place's first and last files: if they hold the gap-fill dates Table 5 describes for Layer 2, that is reported, and any use of them is an amendment.

## Definitions

L1, L2 and L3 are layers 1–3 of one pixel on one day, a pixel-day. Every test is on the native integer codes, with no tolerance. Each pixel-day is exactly one of nodata, fully actual or filled, tested in that order.

| Term | Test |
|---|---|
| nodata | L1, L2 or L3 is −1000 |
| fully actual | not nodata, and L1 = 1: actual PlanetScope and/or Sentinel-2 or Landsat 8 data with no gap filling (Table 5; §3.2) |
| filled | not nodata, and L1 from 2 to 100; entirely filled at 100, blended from 2 to 99 |
| Layer 2 source: same day | L2 is 0, 900 (Sentinel-2 data of the day used to fill) or −900 (Landsat 8 data of the day used to fill), reported by code |
| Layer 2 source: past | L2 from −899 to −1; the offset is L2 days |
| Layer 2 source: future | L2 from 1 to 899; the offset is L2 days |
| Layer 3 status | L3: −999 no PlanetScope data; 2–6 cloud, shadow or haze, also by code; 7 misaligned; 1 marked clear; reported for entirely filled and blended pixel-days apart |
| day class | all: every pixel of the tile fully actual; none: no pixel fully actual; some: neither |
| gap | for one pixel, a maximal run of consecutive days that aren't fully actual; a gap that includes 2018-01-01 or 2019-12-31 is censored, counted and left out of the summaries; so are runs of days in class none |
| window | a set of days before a label date; complete when every day lies in the series |
| covered | a pixel with at least one fully-actual day in the window |

For a filled pixel, Layer 2 gives a signed offset to one source. Table 5 calls it the closest scene with actual data; §4.4 calls the reported tile the one with the highest weight. The specification doesn't reconcile the two, and this plan doesn't assume they're the same tile. Typically up to six tiles from before and after feed a fill (§4.4), and smoothing can draw on the days either side. The measure is the share of filled pixels whose Layer 2 code points to the same day, an earlier date or a later one: a QA indicator, not the share of a fill's information that comes from later dates. Layer 3 is Planet's PlanetScope cloud and shadow mask (Table 5), so its status beside a filled pixel describes the PlanetScope data, not why Fusion filled the pixel.

## What is measured

1. **Real share:** the share of pixel-days fully actual, by place and calendar month and by place over the two years, and the number of days in each day class.
2. **Gaps:** by place, the distribution of gap lengths, weighted by pixel-days (the gap each not-fully-actual pixel-day lies in) and by gaps (each gap once), as the median, 90th percentile and maximum, with censored gaps counted apart; and the runs of consecutive days in class none.
3. **Label dates:** for each of the 96 frames and each of their quarters, the shares fully actual and filled, the shares of filled pixels by Layer 2 source, and the median, 90th-percentile and maximum offsets of past and of future sources.
4. **Windows,** for Post 3: for each cell (place, label date, quarter) and window, the covered share of the quarter's pixels; the 10th, 50th and 90th percentiles of the pixels' numbers of fully-actual days; and the number of days with at least one fully-actual pixel in the quarter. For each complete 90-day cell, the shares of the quarter's pixels covered in 0, 1, 2 and all 3 of the 30-day blocks, since composites from all three blocks need the same pixels covered in each.
5. **Layer 3 status of filled pixels:** the shares of filled pixel-days by Layer 3 status, for entirely filled and blended pixel-days apart, by place and month: a mask status, not a diagnosis of why Fusion filled a pixel.

**Weights.** Pixel-weighted values are primary: a share pools the pixel-days or pixels it summarizes. Date-weighted values (the share within each day, frame or cell, averaged with equal weights) are reported beside them wherever the two round to different whole percentages. Every day has 1,048,576 pixels, so the plain fully-actual share is the same under both weights; its date-level view is the day classes. A share of pixel-days counts every pixel-day it covers, nodata included; a share of filled pixel-days counts filled ones only. Offsets, gap lengths and day counts are summarized by quantiles, pixel-weighted (gaps also with each gap once). A quantile is the smallest observed value whose cumulative weight reaches the given fraction (the inverted CDF), so every quantile reported is an observed whole number of days.

## Windows

Nested windows of 15, 30, 60 and 90 days ending the day before each label date, and three consecutive 30-day blocks: 1–30, 31–60 and 61–90 days before it. A window counts only where it is complete: 23 label dates a place (from 2018-02-01) for the 15- and 30-day windows and the first block, and 21 (from 2018-04-01) for the rest; 2018-01-01 has none. Incomplete windows are listed and left out of every summary. Comparisons across windows use the 21 label dates from 2018-04-01, on which every window is complete; each window's own summary keeps all its complete label dates and shows its denominator.

## Example frame

The label-date frame with the largest share of its pixels filled and with a future Layer 2 source; a tie goes to the earliest date, then the lowest cube number. It is shown as the label date's true color and its Layer 2 map, disclosed as the extreme, with the median of the same share over the 96 frames beside it. The map shows past and future offsets on a diverging scale; 0, 900 and −900 as same-day categories of their own rather than offsets; fully-actual pixels as one category; and nodata masked. Label-date Layer 2 summaries were computed before this plan, so the frame may be foreseeable; the rule picks it, not a look.

## Checks at run time

The run stops if any of these fails:

- 730 files per place, one per day, none duplicated; 9 bands of int16, 1024 × 1024, nodata −1000, one CRS and transform per place.
- Every code within its documented range: L1 1–100, L2 −900 to 900, L3 1–7 or −999, or nodata.
- Counts that partition: fully actual, filled and nodata make up every quarter's 262,144 pixels each day; entirely filled and blended make up filled, and so do the Layer 3 statuses; at the label dates, same day, past and future make up filled. The quarters make up the tile, and the day classes make up 730 days.
- For every pixel, its fully-actual days and the days in its gaps sum to 730.
- Every window has its declared length and ends no later than the day before its label date; the numbers of complete windows are those above; and the 30-day window and the first block, which cover the same days, give identical values.
- For every complete 90-day cell, the pixels covered in 0, 1, 2 and 3 blocks make up the quarter; those covered in at least one block are exactly those the 90-day window covers; and those covered in all three are no more than those covered in any one block.
- At the 96 label dates the run reproduces Post 1's committed QA: 14 frames with every pixel 1, 55 with every pixel 100 and 27 others, and the quarter flags in `cell_qa_status.parquet` row by row (70 fully actual, 237 entirely filled); and L2 = ±900 in 11 frames on 4,757,275 pixels (900: 4,209,777; −900: 547,498).
- The example frame's rule returns one frame.

## Expectations, written before the full run

Each is reported with its observed value, met or not met; none changes a definition. They were written with the exposure set out under *Already seen*, so they're expectations informed by exploratory work, not blind predictions; EX2–EX5 draw on the label dates, which are part of the series measured.

- **EX1.** No pixel-day is nodata. *Basis:* the product is gap-free by design (§4.4).
- **EX2.** Pooled over the four places, 15% to 40% of pixel-days, inclusive, are fully actual. *Basis:* at the label dates the pooled share is between 22% and 34% (14 frames fully actual, 15 more over half, 6 at half or less, 61 with none), and first days of months are a systematic sample of days.
- **EX3.** Andong's fully-actual share is higher than at each other place. *Basis:* 12 of its 24 label-date frames are over half fully actual, against 5, 7 and 5 elsewhere.
- **EX4.** At every place, fewer than a third of days are in class some. *Basis:* 21 of the 96 label-date frames (22%) are partial.
- **EX5.** Gaps are longest at Renshou: its median gap length, weighted by pixel-days, is larger than at each other place. *Basis:* it ties for the fewest mostly-actual label dates and lies in the Sichuan Basin, known for persistent winter cloud and fog.
- **EX6.** Among filled pixel-days, the share with no PlanetScope data is higher at Renshou, and lower at Andong, than at any other place. *Basis:* PlanetScope acquisition frequency over the four places, seen in Planet's catalog in unpublished work.
- **EX7.** For the 30-day window, the median covered share over a place's complete cells is at least 80% at every place. *Basis:* an expectation that fully-actual days are spread out enough, informed by the label-date sample; the overall share alone doesn't imply this coverage, since fully-actual days can cluster.

No expectation is stated for the label dates (estimand 3): their Layer 2 summaries were computed before this plan, so it would not be a test.

## Already seen

- Layer 1 classes of the 96 label-date frames and their 384 quarters, by exact codes (Post 1's `cell_qa_status.parquet` and `frame_endpoints.parquet`), and the frame counts by share fully actual in Post 1's `screen_config.json`.
- Layer 2 at the 96 label dates: per-frame median and 90th-percentile offsets, with same-day codes as zero distance (`frame_qa_summary.parquet`), and the counts of same-day codes.
- In unpublished work: eight label-date frames in detail, by Layer 1–3 categories; PF-QA on at least seven other days, the acquisition dates of PlanetScope scenes examined there; and PlanetScope acquisition frequency over the four places in Planet's catalog. Which frames and days these were comes from Planet's catalog, which this project doesn't publish, so they aren't listed.
- An exploratory analysis of the daily series at four other places, in the Americas: PlanetScope coverage close to all or nothing on each day, and cloudier weather on days without PlanetScope data.
- Not seen: the daily series of these four places, apart from the days above.

## Limits

- **Fully actual isn't causal.** Real pixels are reduced-look-ahead: calibration samples Sentinel-2 and Landsat 8 references from a window typically a year wide, centered on the period processed (§4.3), alignment uses the closest clear days, which can be later ones (§4.5), anomalous pixels may be smoothed with the days either side (§4.4), and a past period is processed in backfill, with data after each date typically available (§4.7).
- **One source per pixel.** The share of filled pixels whose Layer 2 offset points later is a QA indicator, not the share of a fill's information from later dates. Under §4.4's reading, where the reported tile contributes to the fill, the share pointing later is a lower bound on the share of filled pixels with any later-dated input; Table 5's wording alone doesn't establish that.
- **The codes' version.** They follow the March 2021 specification; the run's range and reproduction checks would catch a different encoding of these codes, but don't prove the version.
- **Four places aren't the dataset.** They were chosen for Post 1 by a label-support screen.

## Outputs

Plain CSV tables in `runs/`: `qa_daily.csv` (place × day × quarter, 11,680 rows), `qa_monthly.csv`, `gaps.csv`, `gap_lengths.csv`, `label_frames.csv` (96 rows), `label_cells.csv` (384 rows), `windows.csv`, `block_coverage.csv` (336 rows) and `windows_summary.csv`; `audit.json` with the checks, diagnostics (Layer 2 and Layer 3 codes on fully-actual pixel-days), the files' metadata tags, the code's commit, `plan.json`'s SHA-256 and software versions; `expectations.json`; `inputs.sha256`. Figures in `figures/`: the daily strips, the label-date frames, the example frame and the windows, each with the Fusion attribution. Shares are whole percentages (one decimal below 1%), offsets signed days.
