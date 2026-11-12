"""Fold geometry, and training crops that stay inside the training region."""
import numpy as np

from geoaiwatch.folds import TEST_MASKS, TRAIN_MASKS, train_blocks, train_crop_origin, verify


def test_fold_geometry():
    verify()


def test_training_crops_never_touch_the_test_quarter():
    for fold in TRAIN_MASKS:
        used = set()
        for idx in range(3000):
            r0, c0 = train_crop_origin(fold, np.random.default_rng((0, idx)), 256)
            window = (slice(r0, r0 + 256), slice(c0, c0 + 256))
            assert TRAIN_MASKS[fold][window].all()
            assert not TEST_MASKS[fold][window].any()
            used.add(next(i for i, (a, b, c, d) in enumerate(train_blocks(fold))
                          if a <= r0 < b and c <= c0 < d))
        assert used == {0, 1, 2}, "every training square is used"
