# Design files

These files were fixed before any model was trained. [`METHOD.md`](../METHOD.md) gives the dates and commits.

| File | What it holds |
|---|---|
| `screen_config.json` | the frozen design: candidate classes, support thresholds, expected dates and fold geometry |
| `screen_result.json` | the outcome of the eligibility screen: the retained classes and places |
| `cell_qa_status.parquet` | each of the 384 test cells' gap-fill status, from Planet Fusion's QA band |
| `frame_qa_summary.parquet` | for each of the 96 frames, the observed fraction and the distance in days to the reference observations |
| `frame_endpoints.parquet` | for each frame, whether it is fully observed or fully gap-filled |
| `integrity.parquet`, `support_cells.parquet` | class pixel counts for every label date, by place and by place and quadrant: the screen's input |

`screen_config.json` is kept exactly as frozen, so three of its fields need a note:

- `frozen_commit` holds the placeholder `<sha>`, which was never filled in. The freeze is the private commit `e08a144`.
- `frozen_at` gives the freeze's date at midnight UTC, not its time.
- `secondary_analysis` describes a comparison of whole frames and names a frame list, `observed_frames.parquet`, that isn't shipped. Its counts (29 fully observed frames, 61 fully gap-filled, 6 mixed) came from a threshold at half the frame, not from its stated criterion; applied exactly, the criterion gives 14, 55 and 27, as `frame_endpoints.parquet` records. Before any training (private commit `75e9685`), the comparison was redefined on each test cell's gap-fill status, which `cell_qa_status.parquet` holds.
