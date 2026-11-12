"""Fetch the post's example frame from the DynamicEarthNet record at TUM, not the whole archives.

    python posts/2026-11-near-infrared/scripts/fetch_frame.py           # into DEN_HOME
    python posts/2026-11-near-infrared/scripts/fetch_frame.py --list    # what it would fetch

Reads single files from inside the zip archives over FTP: one Planet Fusion image (about 8 MB)
and the place's 24 monthly labels (about 1 MB each). The data are CC BY-SA 4.0 (Toker et al.,
2022; https://mediatum.ub.tum.de/1650201).
"""
import argparse
import ftplib
import io
import zipfile
from pathlib import Path, PurePosixPath

from geoaiwatch import config as C

HOST, LOGIN = "dataserv.ub.tum.de", "m1650201"          # the record's published login
CUBE, DATE = "6813_3313_13", "2018-04-01"
IMAGERY, LABELS = "planet.50N.zip", "labels.zip"


class FTPRange(io.RawIOBase):
    """A seekable, read-only view of one file on an FTP server, read in byte ranges."""

    def __init__(self, ftp, name):
        self.ftp, self.name, self.pos = ftp, name, 0
        ftp.voidcmd("TYPE I")
        self.size = ftp.size(name)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        conn = self.ftp.transfercmd(f"RETR {self.name}", rest=self.pos)
        got = 0
        with memoryview(buf) as view:
            while got < n:
                chunk = conn.recv(min(1 << 20, n - got))
                if not chunk:
                    break
                view[got:got + len(chunk)] = chunk
                got += len(chunk)
        conn.close()
        try:
            self.ftp.voidresp()                      # 226, or 426 when the range ends early
        except ftplib.Error:
            pass
        self.pos += got
        return got


def reroot(name, keep):
    """The part of an archive path from the first segment that keep() accepts."""
    parts = PurePosixPath(name).parts
    return Path(*parts[next(i for i, p in enumerate(parts) if keep(p)):])


def fetch(ftp, archive, wanted, keep, dest, listing=False):
    with io.BufferedReader(FTPRange(ftp, archive), 1 << 20) as f, zipfile.ZipFile(f) as z:
        names = [n for n in z.namelist() if not n.endswith("/") and wanted(PurePosixPath(n))]
        for n in names:
            out = dest / reroot(n, keep)
            print(f"{archive}: {n} -> {out}")
            if not listing:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(n))
    return names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dest", type=Path, default=C.DEN_HOME)
    p.add_argument("--list", action="store_true")
    p.add_argument("--host", default=HOST)
    p.add_argument("--port", type=int, default=21)
    a = p.parse_args()
    ftp = ftplib.FTP()
    ftp.connect(a.host, a.port, timeout=60)
    ftp.login(LOGIN, LOGIN)
    root = a.dest / "planet"
    image = fetch(ftp, IMAGERY, lambda q: CUBE in q.parts and "PF-SR" in q.parts
                  and q.name == f"{DATE}.tif", lambda s: s == CUBE, root, a.list)
    labels = fetch(ftp, LABELS, lambda q: q.suffix == ".tif"
                   and any(C.strip_zone(s) == CUBE for s in q.parts),
                   lambda s: C.strip_zone(s) == CUBE, root / "labels", a.list)
    ftp.quit()
    print(f"{len(image)} image and {len(labels)} label files"
          + (" found" if a.list else f" under {root}"))
    assert len(image) == 1 and len(labels) == 24, "expected one image and 24 labels"


if __name__ == "__main__":
    main()
