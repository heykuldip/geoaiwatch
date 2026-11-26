"""The frozen plan for the post, read from artifacts/plan.json.

A change after the plan was frozen is an amendment in artifacts/amendments.json; plan.json
itself stays as frozen. Both files are checked against the SHA-256 of the versions this code
implements on every run, so a changed amendment needs a new one and a change to this code.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

POST = Path(__file__).resolve().parents[1]
ART = POST / "artifacts"
PLAN = ART / "plan.json"
AMENDMENTS = ART / "amendments.json"

PLAN_SHA256 = "fafec0e537f03fe3c7e318ae560739ab8c79eeb65a4b3ca1bfc122e1c17b6edf"
AMENDMENTS_SHA256 = "2d73cea352c26afc1fd5132a7cbee2c46c149946d89d546390cd3a552a602295"
IMPLEMENTED = (1,)

INVARIANTS = tuple(f"I{i}" for i in range(1, 14))
EXPECTATIONS = tuple(f"EX{i}" for i in range(1, 8))

# numbers the checks take from the plan's text; tests find each one in plan.json
COMPLETE_DATES = {"short": 23, "long": 21}
COMPLETE_CELLS = {"short": 368, "long": 336}
SHORT_WINDOWS = ("nested_15", "nested_30", "block_1_30")
POST1_FRAMES = {"all_1": 14, "all_100": 55, "other": 27}
POST1_QUARTERS = {"fully_actual": 70, "entirely_filled": 237}
SAME_DAY_CODES = {"frames": 11, "pixels": 4_757_275, 900: 4_209_777, -900: 547_498}
ROWS = {"qa_daily.csv": 11_680, "label_frames.csv": 96, "label_cells.csv": 384,
        "block_coverage.csv": 336}


class PlanError(RuntimeError):
    pass


@dataclass(frozen=True)
class Place:
    cube: str
    aoi: str
    name: str


@dataclass(frozen=True)
class Window:
    """Days `first` to `last` before the label date, both included."""
    name: str
    first: int
    last: int

    @property
    def days(self) -> int:
        return self.first - self.last + 1


@dataclass(frozen=True)
class Design:
    places: tuple[Place, ...]
    first_day: dt.date
    n_days: int
    label_dates: tuple[dt.date, ...]
    tile: int
    quarter: int
    quarters: tuple[tuple[str, int, int], ...]
    bands: int
    dtype: str
    nodata: int
    windows: tuple[Window, ...]

    @property
    def days(self) -> list[dt.date]:
        return [self.first_day + dt.timedelta(days=i) for i in range(self.n_days)]

    @property
    def last_day(self) -> dt.date:
        return self.first_day + dt.timedelta(days=self.n_days - 1)

    def index(self, day: dt.date) -> int:
        return (day - self.first_day).days

    def place(self, name: str) -> Place:
        return next(p for p in self.places if p.name == name)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def month_starts(first: dt.date, last: dt.date) -> list[dt.date]:
    out, d = [], first
    while d <= last:
        out.append(d)
        d = dt.date(d.year + d.month // 12, d.month % 12 + 1, 1)
    return out


def design_from(plan: dict) -> Design:
    """The plan's structured values, with the consistency the code relies on checked."""
    d = plan["data"]
    places = tuple(Place(p["cube"], p["aoi"], p["place"]) for p in d["places"])
    first = dt.date.fromisoformat(d["first_day"])
    last = dt.date.fromisoformat(d["last_day"])
    n_days = (last - first).days + 1
    if n_days != d["days_per_place"]:
        raise PlanError(f"{first}..{last} holds {n_days} days, the plan says {d['days_per_place']}")
    ld = d["label_dates"]
    labels = month_starts(dt.date.fromisoformat(ld["first"]), dt.date.fromisoformat(ld["last"]))
    if len(labels) != ld["per_place"] or any(x.day != 1 for x in labels):
        raise PlanError(f"{len(labels)} label dates, the plan says {ld['per_place']}")
    if labels[0] < first or labels[-1] > last:
        raise PlanError("label dates outside the series")
    per_file = plan["inventory"]["per_file"]
    tile, q = d["tile_px"], d["quarters"]["size_px"]
    if per_file["shape"] != [tile, tile] or per_file["nodata"] != d["nodata"]:
        raise PlanError("the inventory's file shape or nodata differs from the data section")
    quarters = tuple((k, int(r), int(c)) for k, (r, c) in d["quarters"]["origin_row_col"].items())
    windows = tuple(Window(f"nested_{n}", n, 1) for n in plan["windows"]["nested"]["days"])
    blocks = plan["windows"]["blocks"]["days_before"]
    windows += tuple(Window(f"block_{a}_{b}", b, a) for a, b in blocks)
    ids = tuple(x["id"] for x in plan["invariants"])
    if ids != INVARIANTS:
        raise PlanError(f"invariants {ids}, this code checks {INVARIANTS}")
    ids = tuple(x["id"] for x in plan["expectations"])
    if ids != EXPECTATIONS:
        raise PlanError(f"expectations {ids}, this code evaluates {EXPECTATIONS}")
    design = Design(places, first, n_days, tuple(labels), tile, q, quarters, per_file["bands"],
                    per_file["dtype"], d["nodata"], windows)
    check_quarters(design)
    return design


def check_quarters(design: Design) -> None:
    """The quarters cover the tile once each."""
    seen = np.zeros((design.tile, design.tile), np.int64)
    for _, r0, c0 in design.quarters:
        seen[r0:r0 + design.quarter, c0:c0 + design.quarter] += 1
    if seen.shape != (design.tile, design.tile) or not (seen == 1).all():
        raise PlanError("the quarters don't partition the tile")


def read_amendments(path: Path = AMENDMENTS, implemented: tuple = IMPLEMENTED,
                    pinned: str = AMENDMENTS_SHA256) -> list[dict]:
    """The amendments: the file this code implements, byte for byte, holding exactly the
    amendments it implements."""
    path = Path(path)
    if not path.exists():
        if implemented:
            raise PlanError(f"{path.name} is missing; this code implements amendments "
                            f"{list(implemented)}")
        return []
    digest = sha256(path)
    if digest != pinned:
        raise PlanError(f"{path.name} has SHA-256 {digest}, not that of the amendments this "
                        f"code implements, {pinned}")
    doc = json.loads(path.read_text())
    if doc.get("plan_sha256") != PLAN_SHA256:
        raise PlanError(f"{path.name} doesn't name the frozen plan's SHA-256")
    found = doc["amendments"]
    ids = tuple(a.get("id") for a in found)
    if ids != tuple(implemented):
        raise PlanError(f"amendments {list(ids)} in {path.name}; "
                        f"this code implements {list(implemented)}")
    for a in found:
        missing = [k for k in ("date", "results_seen", "reason") if not a.get(k)]
        if missing:
            raise PlanError(f"amendment {a['id']} lacks {missing}")
    return found


def load(plan_path: Path = PLAN, amendments_path: Path = AMENDMENTS) -> tuple[Design, dict]:
    """The design, and a record of what was read: the plan, its SHA-256 and the amendments."""
    raw = Path(plan_path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PLAN_SHA256:
        raise PlanError(f"{plan_path} has SHA-256 {digest}, not the frozen plan's {PLAN_SHA256}")
    plan = json.loads(raw)
    amendments = read_amendments(amendments_path)
    record = {"plan": plan, "plan_sha256": digest, "amendments": amendments,
              "amendments_sha256": sha256(amendments_path) if Path(amendments_path).exists()
              else None}
    return design_from(plan), record
