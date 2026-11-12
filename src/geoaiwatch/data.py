"""Dataset for the RGB vs RGB+NIR comparison.

Both arms see the same crops, flips and order; they differ only in whether the fourth channel
holds near-infrared or zeros. Normalization statistics come from the fold's training
quadrants, and the control channel is zeroed after normalization. The 96 frames are held in
memory for a run.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from .config import BGRN_TO_RGBN, IGNORE_INDEX, SR_SCALE, cube_dir
from .folds import TEST_MASKS, TRAIN_MASKS, train_crop_origin
from .labels import label_paths, read_label, remap_to_retained


# io

def read_sr(path: Path) -> np.ndarray:
    """PF-SR GeoTIFF -> float32 reflectance in RGBN order, shape (4, H, W)."""
    with rasterio.open(path) as s:
        a = s.read()
    assert a.shape[0] == 4, f"{path}: {a.shape[0]} bands"
    a = a[list(BGRN_TO_RGBN)]                      # B,G,R,N -> R,G,B,N
    return a.astype(np.float32) / SR_SCALE


@dataclass(frozen=True)
class Frame:
    cube: str
    date: str
    sr: Path
    label: Path


def build_manifest(cubes: list[str], dates: list[str] | None = None) -> list[Frame]:
    out = []
    for c in cubes:
        d = cube_dir(c)
        labs = label_paths(c)
        for dt, lp in sorted(labs.items()):
            if dates is not None and dt not in dates:
                continue
            sp = d / "PF-SR" / f"{dt}.tif"
            if not sp.exists():
                raise FileNotFoundError(sp)
            out.append(Frame(c, dt, sp, lp))
    return out


# frame cache: loaded once and shared by every dataset built from the same
# manifest; read-only arrays keep forked workers' pages shared

class FrameCache:
    """In-memory PF-SR and 7-class labels for a manifest."""

    def __init__(self, manifest: list[Frame], verbose: bool = True):
        self.manifest = manifest
        self.x: list[np.ndarray] = []
        self.y: list[np.ndarray] = []
        for i, f in enumerate(manifest):
            x = read_sr(f.sr)
            y = read_label(f.label, validate=False)
            x.setflags(write=False)
            y.setflags(write=False)
            self.x.append(x)
            self.y.append(y)
            if verbose and (i + 1) % 24 == 0:
                print(f"  cached {i + 1}/{len(manifest)} frames", flush=True)
        if verbose:
            print(f"FrameCache: {len(manifest)} frames, {self.nbytes / 2**30:.2f} GiB")

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.x) + sum(a.nbytes for a in self.y)

    def __len__(self) -> int:
        return len(self.manifest)


# normalization: training quadrants of the current fold only

def fold_stats(cache: FrameCache, fold: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std over the fold's training pixels."""
    tm = TRAIN_MASKS[fold]
    n = 0
    s1 = np.zeros(4, np.float64)
    s2 = np.zeros(4, np.float64)
    for x in cache.x:
        v = x[:, tm]                                # (4, n_train_px)
        n += v.shape[1]
        s1 += v.sum(axis=1)
        s2 += (v.astype(np.float64) ** 2).sum(axis=1)
    mean = s1 / n
    var = np.maximum(s2 / n - mean ** 2, 1e-12)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


# dataset

class AblationDataset(Dataset):
    """One geometric crop per item, for both arms.

    Pixels outside the fold's training region are set to IGNORE_INDEX, so they add nothing to the
    loss.
    """

    def __init__(
        self,
        cache: FrameCache,
        fold: str,
        retained: list[str],
        mean: np.ndarray,
        std: np.ndarray,
        *,
        train: bool,
        crop: int = 256,
        crops_per_frame: int = 8,
        seed: int = 0,
        augment: bool = True,
    ):
        self.cache = cache
        self.fold = fold
        self.retained = retained
        self.mean = torch.as_tensor(mean).view(4, 1, 1)
        self.std = torch.as_tensor(std).view(4, 1, 1)
        self.train = train
        self.crop = crop
        self.crops_per_frame = crops_per_frame if train else 1
        self.augment = augment and train
        self.seed = seed
        self.region = TRAIN_MASKS[fold] if train else TEST_MASKS[fold]

        # class remapping and region masking are fold-constant, so do them once
        self._y = [
            np.where(self.region,
                     remap_to_retained(y, retained),
                     IGNORE_INDEX).astype(np.int64)
            for y in cache.y
        ]
        for a in self._y:
            a.setflags(write=False)

    def __len__(self) -> int:
        return len(self.cache) * self.crops_per_frame

    def __getitem__(self, idx: int):
        i = idx // self.crops_per_frame
        f = self.cache.manifest[i]
        rng = np.random.default_rng((self.seed, idx))

        x = self.cache.x[i]
        y = self._y[i]

        if self.train:
            r0, c0 = train_crop_origin(self.fold, rng, self.crop)
            assert self.region[r0:r0 + self.crop, c0:c0 + self.crop].all(), (self.fold, r0, c0)
            x = x[:, r0:r0 + self.crop, c0:c0 + self.crop]
            y = y[r0:r0 + self.crop, c0:c0 + self.crop]

        # copy: the cache is read-only and torch needs writable memory
        # copy: an in-place op on a view of the cache would change the shared frame
        x = torch.from_numpy(np.array(x, copy=True))
        y = torch.from_numpy(np.array(y, copy=True))

        # geometric augmentation only, identical across channels and arms
        if self.augment:
            if rng.random() < 0.5:
                x, y = torch.flip(x, [2]), torch.flip(y, [1])
            if rng.random() < 0.5:
                x, y = torch.flip(x, [1]), torch.flip(y, [0])
            k = int(rng.integers(0, 4))
            if k:
                x, y = torch.rot90(x, k, [1, 2]), torch.rot90(y, k, [0, 1])

        x = (x - self.mean) / self.std                  # normalize first

        x_nir = x
        x_null = x.clone()
        x_null[3] = 0.0                                 # then zero the control arm's fourth channel

        return {"rgbn": x_nir, "rgb0": x_null, "y": y,
                "cube": f.cube, "date": f.date}


# contract checks, run on the first batch of every run

def assert_contract(batch: dict, retained: list[str]) -> None:
    a, b, y = batch["rgbn"], batch["rgb0"], batch["y"]

    assert a.shape == b.shape, "arms differ in shape"
    assert a.dtype == b.dtype == torch.float32
    assert a.shape[1] == 4, f"expected 4 channels, got {a.shape[1]}"

    # the null arm's fourth channel is exactly zero, post-normalization
    assert torch.count_nonzero(b[:, 3]) == 0, "null channel is not zero"

    # the arms differ only in channel 3
    assert torch.equal(a[:, :3], b[:, :3]), "arms differ outside channel 3"
    assert torch.count_nonzero(a[:, 3]) > 0, "NIR arm channel is all zero"

    # labels are in range or ignored
    valid = (y >= 0) & (y < len(retained))
    assert bool(((y == IGNORE_INDEX) | valid).all()), "label outside 0..k-1 / ignore"
