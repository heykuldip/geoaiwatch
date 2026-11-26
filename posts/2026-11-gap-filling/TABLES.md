# The tables in `runs/`

What each column of the run's tables holds, for reusing them without reading the code. [`METHOD.md`](METHOD.md) defines the terms (fully actual, filled, entirely filled, blended, censored, covered, complete) and the weights; `scripts/series_run.py` writes the tables.

## Throughout

- **Places:** `cube` is DynamicEarthNet's cube, `place` its name: `6466` Renshou, `6730` Fengcheng, `6813` Yangzhou, `7026` Andong.
- **Dates:** ISO dates. In the label tables and the window tables, `date` is a label date, the first day of a month, which the post calls a map date.
- **Quarters:** `quarter` is one of an image's four 512 × 512 parts, `NW`, `NE`, `SW` or `SE`, of 262,144 pixels each. An image is 1,024 × 1,024 pixels.
- **Counts and shares:** `n_*` columns count pixels, or pixel-days where a row covers days; `share_*` columns are fractions from 0 to 1, with six decimals, not percentages. A share with `dw` in its name is date-weighted: the shares of single days, frames or quarters, averaged with equal weights; the others pool the counts.
- **Blank cells:** a share whose denominator is zero, a quantile with nothing to rank, and a window that isn't complete are left blank, not 0.
- **Quantiles:** `median`, `p10`, `p50` and `p90` are inverted-CDF quantiles, so each is an observed value: the smallest whose cumulative weight reaches the fraction.
- **Layer 2 sources:** `same_day_0`, `same_day_900` and `same_day_minus900` are Layer 2 codes 0, 900 and −900; `past` and `future` are the post's earlier and later sources, codes −899 to −1 and 1 to 899.
- **Layer 3 statuses:** `no_planetscope` (−999), `clear` (1), `code2` to `code6` (cloud, shadow or haze, in the specification's codes), `misaligned` (7); `cloud_shadow_haze` sums `code2` to `code6` and isn't a further status, so don't add it to them.

## `qa_daily.csv`

One row per place, day and quarter. `n_pixels` (262,144), `n_nodata`, `n_fully_actual`, `n_filled` (= `n_entirely_filled` + `n_blended`). `entirely_filled_l3_<status>` and `blended_l3_<status>`: the quarter's entirely filled or blended pixels with that Layer 3 status.

## `qa_monthly.csv`

One row per place and `period`: a month (`2018-01`) or the two years (`2018-01..2019-12`). Use one or the other: the two-year row already sums the months.

- `n_days`; `n_pixel_days`, `n_nodata`, `n_fully_actual`, `n_filled`, `n_entirely_filled`, `n_blended`: pixel-days in the period. `share_fully_actual` = `n_fully_actual` / `n_pixel_days`.
- `days_all`, `days_some`, `days_none`: days on which every pixel of the image, some, or none is fully actual.
- `n_days_with_entirely_filled`, `n_days_with_blended`: days with at least one pixel of that kind.
- For each kind (`entirely_filled`, `blended`) and status: `n_<kind>_l3_<status>`, its pixel-days; `share_<kind>_l3_<status>`, that count over `n_<kind>`; `share_dw_<kind>_l3_<status>`, each day's share averaged over the `n_days_with_<kind>` days that have pixels of that kind.

## `gaps.csv`

One row per place, `level` and `weight`. At level `pixel`, a gap is one pixel's run of consecutive days that aren't fully actual; at level `tile_none`, a run of days on which no pixel of the image is fully actual.

- `weight`: `pixel_days` (each gap weighted by its length) or `gaps` (each gap once) at level `pixel`; `runs` at level `tile_none`.
- `n_uncensored`, `n_censored_start`, `n_censored_end`, `n_censored_both`: the total weight of the gaps in each class of censoring, so pixel-days for `pixel_days` and gaps or runs for the others. A censored gap touches 2018-01-01 (`start`), 2019-12-31 (`end`) or both.
- `median`, `p90`, `max`: gap length in days, from uncensored gaps only, under the row's weight.

## `gap_lengths.csv`

One row per place, `level`, `censored` (`none`, `start`, `end` or `both`) and `length`, for every length that occurs. `n_gaps`: gaps (or runs) of that length. `n_days` = `length` × `n_gaps`: pixel-days at level `pixel`, days at level `tile_none`. A censored gap's `length` is the part inside the two years.

## `label_frames.csv` and `label_cells.csv`

One row per label-date image (place and `date`), and per quarter of one (with `quarter`).

- `n_pixels`, `n_nodata`, `n_fully_actual`, `n_filled`, `n_entirely_filled`, `n_blended`: pixels.
- `n_l2_same_day_0`, `n_l2_same_day_900`, `n_l2_same_day_minus900`, `n_l2_past`, `n_l2_future`: filled pixels by Layer 2 source; they sum to `n_filled`.
- `share_fully_actual`, `share_filled`: of `n_pixels`.
- `share_l2_same_day` (the three same-day codes together), `share_l2_same_day_0`, `share_l2_same_day_900`, `share_l2_same_day_minus900`, `share_l2_past`, `share_l2_future`: of `n_filled`, so blank where nothing is filled.
- `share_future_of_pixels`: `n_l2_future` over all the pixels, `n_pixels`.
- `past_offset_median`, `past_offset_p90`, `past_offset_max`, `future_offset_median`, `future_offset_p90`, `future_offset_max`: how many days away the earlier and the later sources are, weighted by filled pixels. Earlier ones carry a minus sign, so `past_offset_max` = −63 is the farthest source back, 63 days.
- `l1_all_1`, `l1_all_100`: `true` when every pixel is fully actual, or every one entirely filled.
- `n_l2_900_all`, `n_l2_minus900_all`: pixels with Layer 2 code 900 or −900 whatever their Layer 1 value, fully-actual ones included; not part of the filled counts above.

## `label_codes.csv`

One row per label date, quarter and Layer 2 code `l2` that occurs: `n_filled`, the filled pixels with that code.

## `label_summary.csv`

One row: the 96 label-date images pooled. `n_frames` (96) and `n_frames_with_filled` (images with any filled pixel); the counts of `label_frames.csv`, summed. Each `share_<x>` pools those counts, with the same denominators as in `label_frames.csv`; each `share_dw_<x>` averages the images' own shares, over all 96 for `fully_actual` and `filled` and over the `n_frames_with_filled` images for the Layer 2 shares. The offsets as in `label_frames.csv`, pooled. `example_cube`, `example_place`, `example_date`, `example_n_l2_future`, `example_n_pixels`, `example_share_future_of_pixels`: the example image, figure 3. `median_share_future_of_pixels`: the median image's `share_future_of_pixels`, over all 96.

## `windows.csv`

One row per place, label date, quarter and `window`. `nested_15`, `nested_30`, `nested_60` and `nested_90` are the 15 to 90 days before the label date, ending the day before it; `block_1_30`, `block_31_60` and `block_61_90` are days 1–30, 31–60 and 61–90 before it, both ends included (`nested_30` and `block_1_30` are the same days). `days`, `first_day`, `last_day`: the window's length and its first and last dates. `complete`: `true` when every day lies in 2018–2019; otherwise the columns after it are blank, not zero.

- `n_pixels` (262,144); `n_covered`, pixels with at least one fully-actual day in the window; `share_covered` = `n_covered` / `n_pixels`.
- `days_p10`, `days_p50`, `days_p90`: quantiles of the pixels' numbers of fully-actual days in the window.
- `n_days_any`: days in the window on which at least one pixel of the quarter is fully actual.

## `window_days.csv`

One row per place, label date, quarter, `window` and number of fully-actual days, `fully_actual_days`, for complete windows and the numbers that occur: `n_pixels`, the quarter's pixels with exactly that many.

## `block_coverage.csv`

One row per place, label date and quarter, where the 90-day window is complete. `n_blocks_0` to `n_blocks_3`: pixels with a fully-actual day in none, one, two or all three of the 30-day blocks; `share_blocks_0` to `share_blocks_3`, those over `n_pixels`.

## `windows_summary.csv`

One row per place, `window` and set of label dates, `dates`: `own`, the window's complete ones, or `common`, the 21 on which every window is complete (figure 4).

- `n_label_dates`; `n_cells`, quarters times label dates; `n_pixels`, their pixels summed, so each pixel counts once for every label date.
- `n_covered`, `share_covered`: pooled over those pixels. `share_covered_median`: the median quarter's share, a different statistic: Renshou's 15-day window pools 69% (0.688227) while its median quarter is fully covered.
- `days_p10`, `days_p50`, `days_p90`: quantiles of the pixels' fully-actual days, pooled. `n_days_any_median`: the median quarter's `n_days_any`.
- `share_blocks_0` to `share_blocks_3`: for `nested_90` only, `block_coverage.csv` pooled over the same label dates; blank for the other windows.

## Files that aren't tables

`expectations.json`: each expectation, its observed value and whether it was met. `audit.json`: the checks, the code's commit or files, the plan's SHA-256 and the versions. `inputs.sha256`: the SHA-256 of every dataset file read, with its path under `DEN_HOME`.
