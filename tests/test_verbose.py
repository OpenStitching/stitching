import unittest

import numpy as np

from .context import VERBOSE_DIR, Stitcher, test_input


class TestStitcherVerbose(unittest.TestCase):
    def test_verbose(self):
        stitcher = Stitcher()
        panorama = stitcher.stitch_verbose([test_input("weir*")], verbose_dir=VERBOSE_DIR)

        # Check only that the result is correct.
        # Mostly this test is for checking that no error occurs during verbose mode.
        max_image_shape_derivation = 25
        np.testing.assert_allclose(
            panorama.shape[:2], (673, 2636), atol=max_image_shape_derivation
        )


def start_test():
    unittest.main()


if __name__ == "__main__":
    start_test()
