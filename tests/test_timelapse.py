import unittest

import cv2 as cv
import numpy as np

from .context import Stitcher, load_test_img, test_input


class TestImageComposition(unittest.TestCase):
    def test_timelapse(self):
        stitcher = Stitcher(
            timelapse="as_is",
            timelapse_prefix=test_input("timelapse_"),
            crop=False,
        )
        _ = stitcher.stitch([test_input("s?.jpg")])
        frame1 = load_test_img("timelapse_s1.jpg")

        max_image_shape_derivation = 3
        np.testing.assert_allclose(
            frame1.shape[:2], (700, 1811), atol=max_image_shape_derivation
        )

        left = cv.cvtColor(
            frame1[
                :,
                :1300,
            ],
            cv.COLOR_BGR2GRAY,
        )
        right = cv.cvtColor(
            frame1[
                :,
                1300:,
            ],
            cv.COLOR_BGR2GRAY,
        )

        self.assertGreater(cv.countNonZero(left), 800000)
        self.assertEqual(cv.countNonZero(right), 0)


def start_test():
    unittest.main()


if __name__ == "__main__":
    start_test()
