"""The gap-filling post's fetcher: which files it takes from the run's records, how it finds
them in a zone's archive, and that it writes only files with the recorded SHA-256 and leaves
files already in place as they are. An archive built in the test stands in for the record's."""
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest
from gap_filling_fixtures import scripts_on_path

scripts_on_path()
import fetch_example as X  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "posts" / "2026-11-gap-filling" / "runs"
CUBE, DATE = "6813_3313_13", "2018-01-01"
QA = f"planet/50N/30E-149N/{CUBE}/PF-QA/{DATE}.tif"
SR = f"planet/50N/30E-149N/{CUBE}/PF-SR/{DATE}.tif"
DATA = {QA: b"quality layers", SR: b"surface reflectance"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def quiet(_):
    pass


@pytest.fixture
def runs(tmp_path):
    """A runs/ folder whose records name the example frame, among other files."""
    d = tmp_path / "runs"
    d.mkdir()
    with open(d / "label_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_frames", "example_cube", "example_place", "example_date"])
        w.writerow([96, CUBE, "Yangzhou", DATE])
    lines = [f"{sha(b'other')}  planet/50N/30E-149N/{CUBE}/PF-QA/2018-01-02.tif",
             f"{sha(b'other')}  planet/48N/17E-138N/6466_3380_13/PF-QA/{DATE}.tif",
             f"{sha(DATA[QA])}  {QA}", f"{sha(DATA[SR])}  {SR}"]
    (d / "inputs.sha256").write_text("\n".join(lines) + "\n")
    return d


def archive(members: dict) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    buf.seek(0)
    return buf


def in_archive(prefix="planet.50N/"):
    return {prefix + rel.removeprefix("planet/"): data for rel, data in DATA.items()}


def test_the_files_are_the_example_frames_two(runs):
    cube, files = X.wanted(runs)
    assert cube == CUBE and files == {QA: sha(DATA[QA]), SR: sha(DATA[SR])}
    assert X.archive_name(files) == "planet.50N.zip"


def test_the_committed_records_name_yangzhous_first_day():
    cube, files = X.wanted(RUNS)
    audit = json.loads((RUNS / "audit.json").read_text())["example_frame"]
    assert (cube, audit["cube"], audit["date"]) == (CUBE, CUBE, DATE)
    assert sorted(files) == [QA, SR] and files[SR] == audit["pf_sr_check"]["sha256"]
    assert X.archive_name(files) == "planet.50N.zip"


def test_a_missing_record_stops_it(runs):
    (runs / "inputs.sha256").write_text(f"{sha(DATA[QA])}  {QA}\n")
    with pytest.raises(X.FetchFailed, match="not one file of each"):
        X.wanted(runs)


@pytest.mark.parametrize("prefix", ["", "planet.50N/", "DynamicEarthNet/planet/"])
def test_written_at_the_recorded_paths_whatever_the_archives_folders(runs, tmp_path, prefix):
    cube, files = X.wanted(runs)
    members = in_archive(prefix) | {prefix + f"50N/30E-149N/{CUBE}/PF-QA/2018-01-02.tif": b"x"}
    written = X.fetch(archive(members), files, cube, tmp_path / "den", log=quiet)
    assert sorted(written) == sorted(tmp_path / "den" / r for r in files)
    assert all((tmp_path / "den" / r).read_bytes() == DATA[r] for r in files)
    assert not list((tmp_path / "den").rglob("*.part"))


def test_a_different_file_in_the_archive_is_not_written(runs, tmp_path):
    cube, files = X.wanted(runs)
    members = in_archive() | {f"planet.50N/50N/30E-149N/{CUBE}/PF-SR/{DATE}.tif": b"changed"}
    with pytest.raises(X.FetchFailed, match="not the recorded"):
        X.fetch(archive(members), files, cube, tmp_path / "den", log=quiet)
    assert not (tmp_path / "den" / SR).exists()


@pytest.mark.parametrize("extra", [{}, {"copy/6813_3313_13/PF-QA/2018-01-01.tif": b"x"}])
def test_none_or_two_matches_stop_it(runs, tmp_path, extra):
    cube, files = X.wanted(runs)
    members = (in_archive() if extra else {k: v for k, v in in_archive().items()
                                           if "PF-QA" not in k}) | extra
    with pytest.raises(X.FetchFailed, match="matches in the archive"):
        X.fetch(archive(members), files, cube, tmp_path / "den", log=quiet)


def test_listing_writes_nothing(runs, tmp_path):
    cube, files = X.wanted(runs)
    assert X.fetch(archive(in_archive()), files, cube, tmp_path / "den", listing=True,
                   log=quiet) == []
    assert not (tmp_path / "den").exists()


def test_files_in_place_are_kept_and_not_fetched(runs, tmp_path):
    _, files = X.wanted(runs)
    (tmp_path / QA).parent.mkdir(parents=True)
    (tmp_path / QA).write_bytes(DATA[QA])
    assert X.missing(files, tmp_path, log=quiet) == {SR: files[SR]}


def test_a_different_file_in_place_stops_it_and_is_left(runs, tmp_path):
    _, files = X.wanted(runs)
    (tmp_path / QA).parent.mkdir(parents=True)
    (tmp_path / QA).write_bytes(b"something else")
    with pytest.raises(X.FetchFailed, match="left as it is"):
        X.missing(files, tmp_path, log=quiet)
    assert (tmp_path / QA).read_bytes() == b"something else"


class FakeFTP:
    """ftplib.FTP's calls that main() makes, with the archive read from memory."""
    archive = None
    calls = []

    def connect(self, host, port, timeout):
        FakeFTP.calls.append(("connect", host, port))

    def login(self, user, password):
        FakeFTP.calls.append(("login", user))

    def quit(self):
        FakeFTP.calls.append(("quit",))

    def close(self):
        FakeFTP.calls.append(("close",))


class Reader(io.RawIOBase):
    def __init__(self, ftp, name):
        FakeFTP.calls.append(("open", name))
        self.f = io.BytesIO(FakeFTP.archive)

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        return self.f.seek(offset, whence)

    def tell(self):
        return self.f.tell()

    def readinto(self, buf):
        data = self.f.read(len(buf))
        buf[:len(data)] = data
        return len(data)


@pytest.fixture
def fake(monkeypatch, runs):
    FakeFTP.archive, FakeFTP.calls = archive(in_archive()).getvalue(), []
    monkeypatch.setattr(X, "RUNS", runs)
    monkeypatch.setattr(X.ftplib, "FTP", FakeFTP)
    monkeypatch.setattr(X.fetch_frame, "FTPRange", Reader)
    return FakeFTP


def test_main_reads_the_zones_archive_and_writes_both(fake, tmp_path, capsys):
    assert X.main(["--dest", str(tmp_path)]) == 0
    assert ("open", "planet.50N.zip") in fake.calls and fake.calls[-1] == ("quit",)
    assert all((tmp_path / r).read_bytes() == d for r, d in DATA.items())
    assert "2 files written" in capsys.readouterr().out
    fake.calls.clear()
    assert X.main(["--dest", str(tmp_path)]) == 0 and fake.calls == []


def test_main_stops_on_a_different_file_and_says_so(fake, tmp_path, capsys):
    FakeFTP.archive = archive(in_archive() | {f"planet.50N/50N/30E-149N/{CUBE}/PF-QA/"
                                              f"{DATE}.tif": b"changed"}).getvalue()
    assert X.main(["--dest", str(tmp_path)]) == 2
    assert "stopped:" in capsys.readouterr().out and not (tmp_path / QA).exists()
