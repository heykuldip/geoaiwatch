"""PF-QA layers 1 to 3: the inventory, the read, the plan's definitions as integer tests, and
one pass over a place's daily series.

Every test is on the native integer codes. Each file's bytes are read once, hashed, and decoded
from memory, so the hash in runs/inputs.sha256 is that of the bytes measured.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import RasterioError
from rasterio.io import MemoryFile

from geoaiwatch.config import cube_dir

# the plan's definitions, word for word; a test compares them with plan.json
DEFINITIONS = {
    "order": "Each pixel-day is exactly one of nodata, fully actual or filled, tested in that "
             "order.",
    "nodata": "L1 == -1000 or L2 == -1000 or L3 == -1000",
    "fully_actual": "not nodata, and L1 == 1",
    "filled": "not nodata, and 2 <= L1 <= 100",
    "entirely_filled": "L1 == 100",
    "blended": "2 <= L1 <= 99",
    "fill_source.applies_to": "filled pixel-days at the 96 label dates",
    "fill_source.same_day": "L2 in {-900, 0, 900}, reported by code: 0; 900, Sentinel-2 data of "
                            "the day used to fill; -900, Landsat 8 data of the day used to fill",
    "fill_source.past": "-899 <= L2 <= -1, offset L2 days",
    "fill_source.future": "1 <= L2 <= 899, offset L2 days",
    "layer3_status.no_planetscope_data": "L3 == -999",
    "layer3_status.cloud_shadow_haze": "2 <= L3 <= 6, also reported by code",
    "layer3_status.misaligned": "L3 == 7",
    "layer3_status.marked_clear": "L3 == 1",
    "day_class.all": "all 1,048,576 pixels of the tile fully actual",
    "day_class.none": "no pixel of the tile fully actual",
    "day_class.some": "neither",
    "covered": "A pixel with at least one fully-actual day in the window.",
}

SOURCES = ("same_day_0", "same_day_900", "same_day_minus900", "past", "future")
L3_STATUSES = (("no_planetscope", -999), ("clear", 1), ("code2", 2), ("code3", 3),
               ("code4", 4), ("code5", 5), ("code6", 6), ("misaligned", 7))
CENSORING = ("none", "start", "end", "both")
L2_OFFSET, L2_BINS = 900, 1801          # L2 codes -900..900 as bins 0..1800
L3_OFFSET, L3_BINS = 999, 1007          # L3 codes -999..7 as bins 0..1006
DATE_RE = re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})")
STRUCTURE_ONLY = {"IMAGE_STRUCTURE", "DERIVED_SUBDATASETS"}
PF_SR_BANDS = 4                               # blue, green, red, NIR (the specification's Table 3)
UNREADABLE = (RasterioError, OSError)         # rasterio 1.3's RasterioIOError is an OSError only


class CheckFailed(RuntimeError):
    """A check of the plan failed; the run stops and writes nothing."""

    def __init__(self, check: str, problems):
        self.check = check
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        shown = "; ".join(self.problems[:20])
        more = f" (and {len(self.problems) - 20} more)" if len(self.problems) > 20 else ""
        super().__init__(f"{check} failed: {shown}{more}")


# inventory and headers

def name_date(name: str) -> dt.date | None:
    m = DATE_RE.fullmatch(Path(name).stem)
    if m is None:
        return None
    try:
        return dt.date(*map(int, m.groups()))
    except ValueError:
        return None


def files_by_day(folder: Path) -> tuple[dict[dt.date, list[Path]], list[str], list[str]]:
    """GeoTIFFs by the date in their names; other entries; GeoTIFFs without a date."""
    by_day, others, undated = {}, [], []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in (".tif", ".tiff"):
            others.append(f.name)
            continue
        day = name_date(f.name)
        if day is None:
            undated.append(f.name)
        else:
            by_day.setdefault(day, []).append(f)
    return by_day, others, undated


@dataclass
class Inventory:
    qa: dict[str, dict[dt.date, Path]]
    others: dict[str, list[str]]


def inventory(design, planet_root: Path) -> Inventory:
    """I1: one PF-QA file per place and day of the series, and nothing else."""
    problems = []
    qa, others = {}, {}
    days = set(design.days)
    for p in design.places:
        try:
            root = cube_dir(p.cube, planet_root)
        except FileNotFoundError as e:
            problems.append(str(e))
            continue
        if not (root / "PF-QA").is_dir():
            problems.append(f"{p.cube}: no PF-QA folder")
            continue
        by_day, other, undated = files_by_day(root / "PF-QA")
        others[p.cube] = other
        problems += [f"{p.cube}: no date in {n}" for n in undated]
        problems += [f"{p.cube}: {d} is outside the series" for d in sorted(set(by_day) - days)]
        problems += [f"{p.cube}: {d} missing" for d in sorted(days - set(by_day))]
        problems += [f"{p.cube}: {d} in {len(fs)} files: {', '.join(f.name for f in fs)}"
                     for d, fs in sorted(by_day.items()) if len(fs) > 1]
        qa[p.cube] = {d: fs[0] for d, fs in by_day.items() if d in days and len(fs) == 1}
    if problems:
        raise CheckFailed("I1", problems)
    return Inventory(qa, others)


def find_pf_sr(planet_root: Path, cube: str, day: dt.date) -> Path:
    """The one PF-SR file of a place and day, the example frame's picture."""
    try:
        folder = cube_dir(cube, planet_root) / "PF-SR"
    except FileNotFoundError as e:
        raise CheckFailed("pf_sr", str(e)) from e
    fs = files_by_day(folder)[0].get(day, []) if folder.is_dir() else []
    if len(fs) != 1:
        raise CheckFailed("pf_sr", f"{cube} {day}: {len(fs)} PF-SR files")
    return fs[0]


def read_pf_sr(path: Path, design, ref) -> dict:
    """The example frame's PF-SR file, read once: its SHA-256, and a check that it opens, has
    the product's four bands on a tile of the plan's size, and that the red, green and blue
    bands decode. `ref` is the place's PF-QA grid, recorded beside the file's own."""
    try:
        raw = Path(path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        with MemoryFile(raw) as m, m.open() as s:
            bad = []
            if s.count != PF_SR_BANDS:
                bad.append(f"{s.count} bands, not {PF_SR_BANDS}")
            if (s.height, s.width) != (design.tile, design.tile):
                bad.append(f"{s.height} x {s.width} pixels")
            if bad:
                raise CheckFailed("pf_sr", [f"{path.name}: {x}" for x in bad])
            rgb = s.read([3, 2, 1])
            info = {"sha256": digest, "bands": s.count, "dtypes": sorted(set(s.dtypes)),
                    "nodata": s.nodata, "pixels": [s.height, s.width],
                    "crs": str(s.crs), "transform": list(s.transform)[:6],
                    "same_grid_as_pf_qa": bool(s.crs == ref[0] and s.transform == ref[1])}
    except UNREADABLE as e:
        raise CheckFailed("pf_sr", f"{path.name}: unreadable ({e})") from e
    if rgb.shape != (3, design.tile, design.tile):
        raise CheckFailed("pf_sr", f"{path.name}: red, green and blue read as {rgb.shape}")
    return info


def header_problems(src, design, ref=None) -> list[str]:
    """I2 for one open file; `ref` is the place's (CRS, transform)."""
    out = []
    if src.count != design.bands:
        out.append(f"{src.count} bands")
    if set(src.dtypes) != {design.dtype}:
        out.append(f"data types {sorted(set(src.dtypes))}")
    if (src.height, src.width) != (design.tile, design.tile):
        out.append(f"{src.height} x {src.width} pixels")
    if src.nodata is None or any(v is None or v != design.nodata for v in src.nodatavals):
        out.append(f"nodata {src.nodata}")
    if ref is not None:
        if src.crs != ref[0]:
            out.append("a CRS other than the place's first file's")
        if src.transform != ref[1]:
            out.append("a transform other than the place's first file's")
    return out


def metadata(src) -> dict:
    """Metadata domains and tags; the domain GDAL derives from the file's own name is named
    but not copied."""
    ds = {"": src.tags()}
    for ns in src.tag_namespaces():
        derived = ns == "DERIVED_SUBDATASETS"
        ds[ns] = "(derived by GDAL; not recorded)" if derived else src.tags(ns=ns)
    bands = {}
    for b in range(1, src.count + 1):
        tags = {"": src.tags(bidx=b)}
        tags.update({ns: src.tags(bidx=b, ns=ns) for ns in src.tag_namespaces(bidx=b)})
        tags = {ns: t for ns, t in tags.items() if t}
        if tags:
            bands[str(b)] = tags
    extra = sorted(set(ds) - STRUCTURE_ONLY - {""})
    extra += [f"default:{k}" for k in ds[""] if k != "AREA_OR_POINT"]
    extra += [f"band {b}" for b in bands]
    extra += ["band descriptions"] if any(src.descriptions) else []
    return {"dataset": ds, "bands": bands, "descriptions": list(src.descriptions),
            "beyond_image_structure": extra}


def headers(design, inv: Inventory) -> tuple[dict, dict]:
    """I1 (readable) and I2 on every file, before any pixel is read; the metadata of each
    place's first and last files."""
    unreadable, bad, refs, meta = [], [], {}, {}
    ends = (design.days[0], design.days[-1])
    for p in design.places:
        ref = None
        for day in design.days:
            path = inv.qa[p.cube][day]
            try:
                with rasterio.open(path) as s:
                    bad += [f"{p.cube} {path.name}: {x}" for x in header_problems(s, design, ref)]
                    if ref is None:
                        ref = (s.crs, s.transform)
                    if day in ends:
                        meta[f"{p.cube} {path.name}"] = metadata(s)
            except UNREADABLE as e:
                unreadable.append(f"{p.cube} {path.name}: unreadable ({e})")
        refs[p.cube] = ref
    if unreadable:
        raise CheckFailed("I1", unreadable)
    if bad:
        raise CheckFailed("I2", bad)
    return refs, meta


def read_day(path: Path, design, ref) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """One file's bytes, read once: their SHA-256, and layers 1 to 3 decoded from them."""
    try:
        raw = Path(path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        with MemoryFile(raw) as m, m.open() as s:
            bad = header_problems(s, design, ref)
            if bad:
                raise CheckFailed("I2", [f"{path.name}: {x}" for x in bad])
            a = s.read([1, 2, 3])
    except UNREADABLE as e:
        raise CheckFailed("I1", f"{path.name}: unreadable ({e})") from e
    return digest, a[0], a[1], a[2]


# the definitions

def code_problems(l1, l2, l3, nodata: int) -> list[str]:
    """I3: L1 in 1..100, L2 in -900..900, L3 in 1..7 or -999, each or nodata."""
    out = []
    for name, a, ok in (("L1", l1, ((l1 >= 1) & (l1 <= 100)) | (l1 == nodata)),
                        ("L2", l2, ((l2 >= -900) & (l2 <= 900)) | (l2 == nodata)),
                        ("L3", l3, ((l3 >= 1) & (l3 <= 7)) | (l3 == -999) | (l3 == nodata))):
        if not ok.all():
            out.append(f"{name} codes outside its range: {np.unique(a[~ok])[:10].tolist()}")
    return out


@dataclass
class Classes:
    nodata: np.ndarray
    fully_actual: np.ndarray
    filled: np.ndarray
    entirely_filled: np.ndarray
    blended: np.ndarray


def classify(l1, l2, l3, nodata: int) -> Classes:
    """Nodata first, then fully actual, then filled; each mask from the codes themselves."""
    nod = (l1 == nodata) | (l2 == nodata) | (l3 == nodata)
    fa = ~nod & (l1 == 1)
    filled = ~nod & (l1 >= 2) & (l1 <= 100)
    return Classes(nod, fa, filled, filled & (l1 == 100), filled & (l1 >= 2) & (l1 <= 99))


def sources(l2, filled) -> dict[str, np.ndarray]:
    """Layer 2 source of filled pixels: the same day (by code), the past or the future."""
    return {"same_day_0": filled & (l2 == 0),
            "same_day_900": filled & (l2 == 900),
            "same_day_minus900": filled & (l2 == -900),
            "past": filled & (l2 >= -899) & (l2 <= -1),
            "future": filled & (l2 >= 1) & (l2 <= 899)}


def quarter_slices(design) -> dict[str, tuple[slice, slice]]:
    q = design.quarter
    return {k: (slice(r, r + q), slice(c, c + q)) for k, r, c in design.quarters}


def quantile(values, weights, k: int, of: int = 10) -> int | None:
    """The smallest value whose cumulative weight reaches k/of of the total weight (the
    inverted CDF); None when there is no weight."""
    v = np.asarray(values, np.int64)
    w = np.asarray(weights, np.int64)
    keep = w > 0
    v, w = v[keep], w[keep]
    if v.size == 0:
        return None
    order = np.argsort(v, kind="stable")
    v, cw = v[order], np.cumsum(w[order])
    return int(v[int(np.argmax(cw * of >= k * cw[-1]))])


def largest(values, weights) -> int | None:
    v = np.asarray(values, np.int64)[np.asarray(weights, np.int64) > 0]
    return int(v.max()) if v.size else None


# one day

def day_counts(c: Classes, l3, qs) -> dict[str, dict[str, int]]:
    """Each quarter's counts, each from its own mask."""
    out = {}
    k3 = l3.astype(np.int32) + L3_OFFSET
    for name, sl in qs.items():
        row = {"n_pixels": int(c.nodata[sl].size),
               "n_nodata": int(c.nodata[sl].sum()),
               "n_fully_actual": int(c.fully_actual[sl].sum()),
               "n_filled": int(c.filled[sl].sum()),
               "n_entirely_filled": int(c.entirely_filled[sl].sum()),
               "n_blended": int(c.blended[sl].sum())}
        for kind, mask in (("entirely_filled", c.entirely_filled), ("blended", c.blended)):
            h = np.bincount(k3[sl][mask[sl]], minlength=L3_BINS)
            for status, code in L3_STATUSES:
                row[f"{kind}_l3_{status}"] = int(h[code + L3_OFFSET])
        out[name] = row
    return out


def check_day(counts: dict, c: Classes, design, where: str) -> None:
    """I4 for each quarter and I5 for the tile."""
    q2 = design.quarter ** 2
    bad = []
    for name, r in counts.items():
        if r["n_fully_actual"] + r["n_filled"] + r["n_nodata"] != q2 or r["n_pixels"] != q2:
            bad.append(f"{where} {name}: fully actual, filled and nodata don't make {q2:,}")
        if r["n_entirely_filled"] + r["n_blended"] != r["n_filled"]:
            bad.append(f"{where} {name}: entirely filled and blended don't make filled")
        for kind in ("entirely_filled", "blended"):
            if sum(r[f"{kind}_l3_{s}"] for s, _ in L3_STATUSES) != r[f"n_{kind}"]:
                bad.append(f"{where} {name}: the Layer 3 statuses don't make {kind}")
    if bad:
        raise CheckFailed("I4", bad)
    tile = {"n_nodata": c.nodata, "n_fully_actual": c.fully_actual, "n_filled": c.filled,
            "n_entirely_filled": c.entirely_filled, "n_blended": c.blended}
    for key, mask in tile.items():
        if sum(r[key] for r in counts.values()) != int(mask.sum()):
            bad.append(f"{where}: the quarters' {key} isn't the tile's")
    if sum(r["n_pixels"] for r in counts.values()) != design.tile ** 2:
        bad.append(f"{where}: the quarters don't make the tile's {design.tile ** 2:,} pixels")
    if bad:
        raise CheckFailed("I5", bad)


# gaps

class Gaps:
    """Each pixel's runs of days that aren't fully actual, as histograms of their lengths by
    censoring: none; start (the run includes the first day); end (the last day); both."""

    def __init__(self, shape, n_days: int):
        self.n_days = n_days
        self.run = np.zeros(shape, np.int32)
        self.fully_actual_days = np.zeros(shape, np.int32)
        self.gap_days = np.zeros(shape, np.int32)
        self.hist = {k: np.zeros(n_days + 1, np.int64) for k in CENSORING}
        self.day = 0

    def _add(self, kind: str, lengths) -> None:
        self.hist[kind] += np.bincount(lengths, minlength=self.n_days + 1)

    def add(self, fa) -> None:
        ended = fa & (self.run > 0)
        if ended.any():
            lengths = self.run[ended]
            from_start = lengths == self.day
            self._add("start", lengths[from_start])
            self._add("none", lengths[~from_start])
            self.gap_days[ended] += lengths
        self.run[fa] = 0
        self.run[~fa] += 1
        self.fully_actual_days += fa
        self.day += 1

    def finish(self) -> None:
        if self.day != self.n_days:
            raise CheckFailed("I7", f"{self.day} days added, not {self.n_days}")
        open_ = self.run > 0
        lengths = self.run[open_]
        whole = lengths == self.n_days
        self._add("both", lengths[whole])
        self._add("end", lengths[~whole])
        self.gap_days[open_] += lengths

    def check(self, where: str) -> None:
        """I7: for every pixel, its fully-actual days and the days in its gaps make the series."""
        bad = (self.fully_actual_days + self.gap_days) != self.n_days
        if bad.any():
            raise CheckFailed("I7", f"{where}: {int(bad.sum())} pixels whose fully-actual days "
                                    f"and gap days don't make {self.n_days}")


def runs_of(flags, n_days: int) -> dict[str, np.ndarray]:
    """Runs of consecutive True days, as histograms of their lengths by censoring."""
    hist = {k: np.zeros(n_days + 1, np.int64) for k in CENSORING}
    i = 0
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue
        j = i
        while j < len(flags) and flags[j]:
            j += 1
        kind = {(True, True): "both", (True, False): "start", (False, True): "end",
                (False, False): "none"}[(i == 0, j == len(flags))]
        hist[kind][j - i] += 1
        i = j
    return hist


# windows

class Ring:
    """The fully-actual flags of the days before the current one, as many as the longest
    window spans."""

    def __init__(self, depth: int, shape):
        self.depth = depth
        self.flags = np.zeros((depth, *shape), bool)
        self.next = 0

    def push(self, t: int, fa) -> None:
        if t != self.next:
            raise CheckFailed("I8", f"day {t} pushed after day {self.next - 1}")
        self.flags[t % self.depth] = fa
        self.next += 1

    def window_days(self, t: int, w) -> list[int]:
        return [t - k for k in range(w.last, w.first + 1)]

    def counts(self, t: int, w) -> np.ndarray:
        """Each pixel's number of fully-actual days in window w before day t, from its own days."""
        days = self.window_days(t, w)
        if self.next != t or min(days) < 0 or min(days) < t - self.depth or max(days) >= t:
            raise CheckFailed("I8", f"{w.name} before day {t}: days {min(days)}..{max(days)}")
        out = np.zeros(self.flags.shape[1:], np.int16)
        for d in days:
            out += self.flags[d % self.depth]
        return out

    def days_with_any(self, t: int, w, sl) -> int:
        return sum(int(self.flags[d % self.depth][sl].any()) for d in self.window_days(t, w))


def window_cells(ring: Ring, t: int, day: dt.date, design, qs) -> tuple[list, list, list]:
    """Estimand 4 for one label date: rows for windows.csv, window_days.csv and
    block_coverage.csv, with I12 and I13 checked."""
    rows, hist_rows, block_rows = [], [], []
    covered = {}
    counts_by_window = {}
    for w in design.windows:
        days = ring.window_days(t, w)
        first, last = day - dt.timedelta(days=w.first), day - dt.timedelta(days=w.last)
        base = {"date": day, "window": w.name, "days": w.days, "first_day": first,
                "last_day": last}
        if len(days) != w.days or last >= day:
            raise CheckFailed("I8", f"{w.name} before {day}: {len(days)} days ending {last}")
        if min(days) < 0:
            rows += [dict(base, quarter=name, complete=False) for name in qs]
            continue
        cnt = ring.counts(t, w)
        counts_by_window[w.name] = cnt
        covered[w.name] = cnt > 0
        for name, sl in qs.items():
            h = np.bincount(cnt[sl].ravel(), minlength=w.days + 1)
            values = np.arange(h.size)
            rows.append(dict(base, quarter=name, complete=True, n_pixels=int(h.sum()),
                             n_covered=int(covered[w.name][sl].sum()),
                             days_p10=quantile(values, h, 1), days_p50=quantile(values, h, 5),
                             days_p90=quantile(values, h, 9),
                             n_days_any=ring.days_with_any(t, w, sl)))
            hist_rows += [{"date": day, "quarter": name, "window": w.name,
                           "fully_actual_days": int(k), "n_pixels": int(n)}
                          for k, n in enumerate(h) if n]
    check_same_window(rows, counts_by_window, day)
    blocks = [w.name for w in design.windows if w.name.startswith("block_")]
    nested_90 = f"nested_{max(w.first for w in design.windows if w.name.startswith('nested_'))}"
    if all(b in covered for b in blocks) and nested_90 in covered:
        k = sum(covered[b].astype(np.int8) for b in blocks)
        bad = []
        if not np.array_equal(k >= 1, covered[nested_90]):
            bad.append(f"{day}: pixels covered in some block aren't those {nested_90} covers")
        for name, sl in qs.items():
            n = np.bincount(k[sl].ravel(), minlength=len(blocks) + 1)
            per_block = [int(covered[b][sl].sum()) for b in blocks]
            if int(n.sum()) != design.quarter ** 2:
                bad.append(f"{day} {name}: 0 to {len(blocks)} blocks don't make the quarter")
            if n[len(blocks)] > min(per_block):
                bad.append(f"{day} {name}: more pixels in all blocks than in one")
            block_rows.append({"date": day, "quarter": name, "n_pixels": int(n.sum()),
                               **{f"n_blocks_{i}": int(n[i]) for i in range(len(n))}})
        if bad:
            raise CheckFailed("I13", bad)
    return rows, hist_rows, block_rows


def check_same_window(rows: list, counts: dict, day: dt.date) -> None:
    """I12: the nested 30-day window and block 1-30 cover the same days and give the same values."""
    a, b = "nested_30", "block_1_30"
    if (a in counts) != (b in counts):
        raise CheckFailed("I12", f"{day}: {a} and {b} differ in completeness")
    if a not in counts:
        return
    if not np.array_equal(counts[a], counts[b]):
        raise CheckFailed("I12", f"{day}: {a} and {b} count different days")
    keys = ("days", "first_day", "last_day", "n_pixels", "n_covered", "days_p10", "days_p50",
            "days_p90", "n_days_any")
    ra = {r["quarter"]: [r[k] for k in keys] for r in rows if r["window"] == a}
    rb = {r["quarter"]: [r[k] for k in keys] for r in rows if r["window"] == b}
    if ra != rb:
        raise CheckFailed("I12", f"{day}: {a} and {b} give different values")


# label dates

def label_frame(day, l1, l2, c: Classes, counts: dict, qs) -> tuple[dict, list, list]:
    """Estimand 3 for one label date: the frame, its quarters, and the filled pixels by
    Layer 2 code; with I4's sources check."""
    src = sources(l2, c.filled)
    cells, codes = [], []
    for name, sl in qs.items():
        r = {k: counts[name][k] for k in ("n_pixels", "n_nodata", "n_fully_actual", "n_filled",
                                          "n_entirely_filled", "n_blended")}
        r.update({f"n_l2_{s}": int(src[s][sl].sum()) for s in SOURCES})
        if sum(r[f"n_l2_{s}"] for s in SOURCES) != r["n_filled"]:
            raise CheckFailed("I4", f"{day} {name}: same day, past and future don't make filled")
        h = np.bincount(l2[sl][c.filled[sl]].astype(np.int32) + L2_OFFSET, minlength=L2_BINS)
        r.update(l1_all_1=bool((l1[sl] == 1).all()), l1_all_100=bool((l1[sl] == 100).all()),
                 n_l2_900_all=int((l2[sl] == 900).sum()),
                 n_l2_minus900_all=int((l2[sl] == -900).sum()), hist=h)
        cells.append(dict(r, date=day, quarter=name))
        codes += [{"date": day, "quarter": name, "l2": int(i - L2_OFFSET), "n_filled": int(n)}
                  for i, n in enumerate(h) if n]
    frame = {k: sum(x[k] for x in cells) for k in cells[0]
             if k.startswith("n_") and not k.endswith("_all")}
    frame.update(date=day, l1_all_1=bool((l1 == 1).all()), l1_all_100=bool((l1 == 100).all()),
                 n_l2_900_all=int((l2 == 900).sum()), n_l2_minus900_all=int((l2 == -900).sum()),
                 hist=sum(x["hist"] for x in cells))
    return frame, cells, codes


def offsets(hist) -> dict:
    """Median, 90th percentile and largest distance of past and of future sources, pixel-
    weighted; past ones reported with a minus sign."""
    codes = np.arange(L2_BINS) - L2_OFFSET
    out = {}
    for side, keep, sign in (("past", (codes >= -899) & (codes <= -1), -1),
                             ("future", (codes >= 1) & (codes <= 899), 1)):
        dist, w = np.abs(codes[keep]), np.asarray(hist)[keep]
        for key, value in (("median", quantile(dist, w, 5)), ("p90", quantile(dist, w, 9)),
                           ("max", largest(dist, w))):
            out[f"{side}_offset_{key}"] = None if value is None else sign * value
    return out


# one place

@dataclass
class PlaceResult:
    cube: str
    daily: list = field(default_factory=list)
    tile_fully_actual: list = field(default_factory=list)
    gaps: dict = field(default_factory=dict)
    frames: list = field(default_factory=list)
    cells: list = field(default_factory=list)
    codes: list = field(default_factory=list)
    windows: list = field(default_factory=list)
    window_days: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    hashes: list = field(default_factory=list)
    seconds: float = 0.0


def place_pass(design, place, files: dict, ref, log=print) -> PlaceResult:
    """One pass over a place's days in date order, each file read once."""
    t0 = time.time()
    shape = (design.tile, design.tile)
    qs = quarter_slices(design)
    res = PlaceResult(place.cube)
    gaps = Gaps(shape, design.n_days)
    ring = Ring(max(w.first for w in design.windows), shape)
    labels = set(design.label_dates)
    l2_fa = np.zeros(L2_BINS, np.int64)
    l3_fa = np.zeros(L3_BINS, np.int64)
    for t, day in enumerate(design.days):
        path = files[day]
        digest, l1, l2, l3 = read_day(path, design, ref)
        res.hashes.append((digest, path))
        bad = code_problems(l1, l2, l3, design.nodata)
        if bad:
            raise CheckFailed("I3", [f"{place.cube} {path.name}: {x}" for x in bad])
        c = classify(l1, l2, l3, design.nodata)
        counts = day_counts(c, l3, qs)
        check_day(counts, c, design, f"{place.cube} {day}")
        res.daily += [dict(r, date=day, quarter=name) for name, r in counts.items()]
        res.tile_fully_actual.append(int(c.fully_actual.sum()))
        if day in labels:
            frame, cells, codes = label_frame(day, l1, l2, c, counts, qs)
            res.frames.append(frame)
            res.cells += cells
            res.codes += codes
            rows, hist_rows, block_rows = window_cells(ring, t, day, design, qs)
            res.windows += rows
            res.window_days += hist_rows
            res.blocks += block_rows
        ring.push(t, c.fully_actual)
        gaps.add(c.fully_actual)
        l2_fa += np.bincount(l2[c.fully_actual].astype(np.int32) + L2_OFFSET, minlength=L2_BINS)
        l3_fa += np.bincount(l3[c.fully_actual].astype(np.int32) + L3_OFFSET, minlength=L3_BINS)
        if log and (t + 1) % 73 == 0:
            log(f"  {place.cube}: {t + 1}/{design.n_days} days read, {time.time() - t0:.0f} s")
    gaps.finish()
    gaps.check(place.cube)
    res.gaps = gaps.hist
    res.diagnostics = {
        "l2_codes_on_fully_actual": {str(i - L2_OFFSET): int(n) for i, n in enumerate(l2_fa) if n},
        "l3_codes_on_fully_actual": {str(i - L3_OFFSET): int(n) for i, n in enumerate(l3_fa) if n}}
    res.seconds = time.time() - t0
    return res
