"""Spatial folds: four rotating quadrants of a 1024 x 1024 tile.

The test quadrant is 512 x 512 and untrimmed. The three training quadrants are each recessed
TRAIN_RECESS_PX from the shared central boundaries, 3 x 480^2 = 691,200 px per image.
"""
from __future__ import annotations

import numpy as np

from .config import HALF, QUADS, TILE, TRAIN_RECESS_PX


def test_mask(fold: str, tile: int = TILE) -> np.ndarray:
    """The held-out quadrant, untrimmed."""
    r0, c0 = QUADS[fold]
    m = np.zeros((tile, tile), bool)
    m[r0:r0 + HALF, c0:c0 + HALF] = True
    return m


def train_mask(fold: str, tile: int = TILE, recess: int = TRAIN_RECESS_PX) -> np.ndarray:
    """The other three quadrants, each recessed from the shared central edges."""
    m = np.zeros((tile, tile), bool)
    for q, (r0, c0) in QUADS.items():
        if q == fold:
            continue
        sub = np.zeros((tile, tile), bool)
        sub[r0:r0 + HALF, c0:c0 + HALF] = True
        if r0 == 0:
            sub[HALF - recess:HALF, :] = False
        else:
            sub[HALF:HALF + recess, :] = False
        if c0 == 0:
            sub[:, HALF - recess:HALF] = False
        else:
            sub[:, HALF:HALF + recess] = False
        m |= sub
    return m


def train_blocks(fold: str, recess: int = TRAIN_RECESS_PX) -> list[tuple[int, int, int, int]]:
    """The training region's three squares, as (row0, row1, col0, col1), ends exclusive."""
    out = []
    for q, (r0, c0) in QUADS.items():
        if q == fold:
            continue
        rows = (r0, HALF - recess) if r0 == 0 else (HALF + recess, r0 + HALF)
        cols = (c0, HALF - recess) if c0 == 0 else (HALF + recess, c0 + HALF)
        out.append((*rows, *cols))
    return out


def train_crop_origin(fold: str, rng: np.random.Generator, crop: int) -> tuple[int, int]:
    """Top-left corner of a training crop that lies wholly inside one training square."""
    r0, r1, c0, c1 = train_blocks(fold)[int(rng.integers(0, 3))]
    return int(rng.integers(r0, r1 - crop + 1)), int(rng.integers(c0, c1 - crop + 1))


TEST_MASKS = {q: test_mask(q) for q in QUADS}
TRAIN_MASKS = {q: train_mask(q) for q in QUADS}


def verify() -> None:
    """Invariants. 3 * 480^2 = 691,200 training px; masks disjoint; no overlap."""
    expect = 3 * (HALF - TRAIN_RECESS_PX) ** 2
    for q in QUADS:
        tr, te = TRAIN_MASKS[q], TEST_MASKS[q]
        assert tr.sum() == expect, f"{q}: train {tr.sum()} != {expect}"
        assert te.sum() == HALF ** 2, f"{q}: test {te.sum()}"
        assert not (tr & te).any(), f"{q}: train/test overlap"
        blocks = np.zeros((TILE, TILE), bool)
        for r0, r1, c0, c1 in train_blocks(q):
            blocks[r0:r1, c0:c1] = True
        assert (blocks == tr).all(), f"{q}: training squares differ from the mask"
    # the four test quadrants exactly partition the tile
    total = np.zeros((TILE, TILE), bool)
    for q in QUADS:
        assert not (total & TEST_MASKS[q]).any()
        total |= TEST_MASKS[q]
    assert total.all(), "test quadrants do not partition the tile"


if __name__ == "__main__":
    verify()
    print(f"train px/image: {TRAIN_MASKS['NW'].sum():,}")
    print(f"test  px/image: {TEST_MASKS['NW'].sum():,}")
    print("fold geometry OK")
