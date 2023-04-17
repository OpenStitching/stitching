import os
import unittest

import cv2 as cv
import numpy as np

from .context import Stitcher, testimg, OUT_DIR


class TestImageComposition(unittest.TestCase):

    def test_timelapse(self):
        stitcher = Stitcher(timelapse="as_is", timelapse_prefix=os.path.join(OUT_DIR, "timelapse_"), crop=False)
        _ = stitcher.stitch([testimg("s?.jpg")])
        frame1 = cv.imread(os.path.join(OUT_DIR, "timelapse_s1.jpg"))

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


def starttest():
    unittest.main()


if __name__ == "__main__":
    starttest()
