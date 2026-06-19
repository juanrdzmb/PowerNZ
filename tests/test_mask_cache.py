import numpy as np

from mask_cache import MaskFrameCache


def test_mask_cache_round_trips_lossless_mask() -> None:
    mask = np.zeros((32, 48), dtype=np.uint8)
    mask[8:20, 10:30] = 255
    with MaskFrameCache() as cache:
        cache.put(4, mask)
        restored = cache.get(4)
        assert restored is not None
        assert np.array_equal(restored, mask)
