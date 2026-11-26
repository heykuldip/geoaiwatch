"""Fetch the post's example frame, its quality file and its image, from the DynamicEarthNet
record at TUM, without the whole archive.

    python posts/2026-11-gap-filling/scripts/fetch_example.py           # into DEN_HOME
    python posts/2026-11-gap-filling/scripts/fetch_example.py --list    # what it would fetch

The frame is the one runs/label_summary.csv names. Its PF-QA and PF-SR files are read from
inside the zone's zip archive over FTP, with the near-infrared post's reader, checked against the
SHA-256 runs/inputs.sha256 records for them, and only then written under DEN_HOME, at the paths
inputs.sha256 gives. A file already in place with the recorded SHA-256 is not fetched again; one
with another SHA-256 is left as it is, and the fetch stops. The data are CC BY-SA 4.0
(Toker et al., 2022; https://mediatum.ub.tum.de/1650201).
"""
from __future__ import annotations

import argparse
import csv
import ftplib
import hashlib
import io
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath

from geoaiwatch import config

POST = Path(__file__).resolve().parents[1]
RUNS = POST / "runs"
sys.path.insert(0, str(POST.parent / "2026-11-near-infrared" / "scripts"))
import fetch_frame  # noqa: E402

PRODUCTS = ("PF-QA", "PF-SR")
ZONE = re.compile(r"\d{2}[NS]")


class FetchFailed(RuntimeError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def wanted(runs: Path = RUNS) -> tuple[str, dict[str, str]]:
    """The example frame's cube, and its two files: {path under DEN_HOME: recorded SHA-256}."""
    with open(Path(runs) / "label_summary.csv", newline="") as f:
        row = next(csv.DictReader(f))
    cube, date = row["example_cube"], row["example_date"]
    files = {}
    for line in (Path(runs) / "inputs.sha256").read_text().splitlines():
        digest, _, rel = line.partition("  ")
        p = PurePosixPath(rel)
        if cube in p.parts and p.parent.name in PRODUCTS and p.stem.replace("_", "-") == date:
            files[rel] = digest
    found = sorted(PurePosixPath(r).parent.name for r in files)
    if found != sorted(PRODUCTS):
        raise FetchFailed(f"{cube} {date}: inputs.sha256 holds {found}, not one file of each "
                          f"of {list(PRODUCTS)}")
    return cube, files


def archive_name(files: dict) -> str:
    """The zone's archive, planet.<zone>.zip, from the zone in the files' paths."""
    zones = {next((s for s in PurePosixPath(r).parts if ZONE.fullmatch(s)), None) for r in files}
    if len(zones) != 1 or None in zones:
        raise FetchFailed(f"no single zone in {sorted(files)}")
    return f"planet.{zones.pop()}.zip"


def missing(files: dict, dest: Path, log=print) -> dict:
    """The files not yet under dest; a file there with another SHA-256 stops the fetch."""
    todo = {}
    for rel, digest in files.items():
        p = Path(dest) / rel
        if not p.exists():
            todo[rel] = digest
            continue
        got = sha256(p.read_bytes())
        if got != digest:
            raise FetchFailed(f"{p} has SHA-256 {got}, not the recorded {digest}; left as it is")
        log(f"{p}: already in place, SHA-256 as recorded")
    return todo


def fetch(archive, files: dict, cube: str, dest: Path, listing: bool = False,
          log=print) -> list[Path]:
    """Read the files from an open zip archive, any seekable binary file, into dest. Each is
    found by its path from the cube's folder on, checked against its SHA-256, then written."""
    written = []
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        for rel, digest in files.items():
            parts = PurePosixPath(rel).parts
            tail = parts[parts.index(cube):]
            hits = [m for m in members if PurePosixPath(m.filename).parts[-len(tail):] == tail]
            if len(hits) != 1:
                raise FetchFailed(f"{'/'.join(tail)}: {len(hits)} matches in the archive")
            out = Path(dest) / rel
            log(f"{hits[0].filename}, {hits[0].file_size / 2**20:.1f} MB -> {out}")
            if listing:
                continue
            data = z.read(hits[0])
            got = sha256(data)
            if got != digest:
                raise FetchFailed(f"{hits[0].filename}: SHA-256 {got}, not the recorded {digest}")
            out.parent.mkdir(parents=True, exist_ok=True)
            part = out.with_name(out.name + ".part")
            part.write_bytes(data)
            part.replace(out)
            written.append(out)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", type=Path, default=config.DEN_HOME)
    ap.add_argument("--list", action="store_true", help="say what it would fetch, writing nothing")
    ap.add_argument("--host", default=fetch_frame.HOST)
    ap.add_argument("--port", type=int, default=21)
    a = ap.parse_args(argv)
    try:
        cube, files = wanted(RUNS)
        todo = missing(files, a.dest)
        if not todo:
            print(f"both files already under {a.dest}")
            return 0
        archive = archive_name(todo)
        ftp = ftplib.FTP()
        ftp.connect(a.host, a.port, timeout=60)
        ftp.login(fetch_frame.LOGIN, fetch_frame.LOGIN)
        try:
            with io.BufferedReader(fetch_frame.FTPRange(ftp, archive), 1 << 20) as f:
                written = fetch(f, todo, cube, a.dest, a.list)
        finally:
            try:
                ftp.quit()
            except ftplib.all_errors:
                ftp.close()
    except FetchFailed as e:
        print(f"stopped: {e}")
        return 2
    print(f"{len(todo)} files found in {archive}" if a.list
          else f"{len(written)} files written under {a.dest}, each with the recorded SHA-256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
