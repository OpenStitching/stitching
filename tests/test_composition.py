import unittest
import os

import numpy as np
import cv2 as cv

from .context import Stitcher


class TestImageComposition(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        TEST_DIR = os.path.abspath(os.path.dirname(__file__))
        os.chdir(os.path.join(TEST_DIR, "testdata"))

    #  visual test: look especially in the sky
    def test_exposure_compensation(self):
        img = cv.imread("s1.jpg")
        img = increase_brightness(img, value=25)
        cv.imwrite("s1_bright.jpg", img)

        stitcher = Stitcher(compensator="no", blender_type="no", crop=False)
        result = stitcher.stitch(["s1_bright.jpg", "s2.jpg"])

        cv.imwrite("without_exposure_comp.jpg", result)

        stitcher = Stitcher(blender_type="no")
        result = stitcher.stitch(["s1_bright.jpg", "s2.jpg"])

        cv.imwrite("with_exposure_comp.jpg", result)

    def test_timelapse(self):
        stitcher = Stitcher(timelapse='as_is', crop=False)
        _ = stitcher.stitch(["s1.jpg", "s2.jpg"])
        frame1 = cv.imread("fixed_s1.jpg")

        max_image_shape_derivation = 3
        np.testing.assert_allclose(frame1.shape[:2],
                                   (700, 1811),
                                   atol=max_image_shape_derivation)

        left = cv.cvtColor(frame1[:, :1300, ], cv.COLOR_BGR2GRAY)
        right = cv.cvtColor(frame1[:, 1300:, ], cv.COLOR_BGR2GRAY)

        self.assertGreater(cv.countNonZero(left), 800000)
        self.assertEqual(cv.countNonZero(right), 0)


def increase_brightness(img, value=30):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return img


def starttest():
    unittest.main()


if __name__ == "__main__":
    starttest()
