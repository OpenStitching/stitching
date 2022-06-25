import os
import unittest

import cv2 as cv
import numpy as np

from .context import AffineStitcher, Stitcher


class TestStitcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEST_DIR = os.path.abspath(os.path.dirname(__file__))
        os.chdir(os.path.join(TEST_DIR, "testdata"))

    def test_stitcher_aquaduct(self):
        stitcher = Stitcher(nfeatures=250, crop=False)
        result = stitcher.stitch(["s1.jpg", "s2.jpg"])
        cv.imwrite("result.jpg", result)

        max_image_shape_derivation = 3
        np.testing.assert_allclose(
            result.shape[:2], (700, 1811), atol=max_image_shape_derivation
        )

    @unittest.skip("skip boat test (high resuolution ran >30s)")
    def test_stitcher_boat1(self):
        settings = {
            "warper_type": "fisheye",
            "wave_correct_kind": "no",
            "finder": "dp_colorgrad",
            "compensator": "no",
            "confidence_threshold": 0.3,
            "crop": False,
        }

        stitcher = Stitcher(**settings)
        result = stitcher.stitch(
            [
                "boat5.jpg",
                "boat2.jpg",
                "boat3.jpg",
                "boat4.jpg",
                "boat1.jpg",
                "boat6.jpg",
            ]
        )

        cv.imwrite("boat_fisheye.jpg", result)

        max_image_shape_derivation = 600
        np.testing.assert_allclose(
            result.shape[:2], (14488, 7556), atol=max_image_shape_derivation
        )

    @unittest.skip("skip boat test (high resuolution ran >30s)")
    def test_stitcher_boat2(self):
        settings = {
            "warper_type": "compressedPlaneA2B1",
            "finder": "dp_colorgrad",
            "compensator": "channel_blocks",
            "confidence_threshold": 0.3,
            "crop": False,
        }

        stitcher = Stitcher(**settings)
        result = stitcher.stitch(
            [
                "boat5.jpg",
                "boat2.jpg",
                "boat3.jpg",
                "boat4.jpg",
                "boat1.jpg",
                "boat6.jpg",
            ]
        )

        cv.imwrite("boat_plane.jpg", result)

        max_image_shape_derivation = 600
        np.testing.assert_allclose(
            result.shape[:2], (7400, 12340), atol=max_image_shape_derivation
        )

    def test_stitcher_boat_aquaduct_subset(self):
        settings = {"final_megapix": 1}

        stitcher = Stitcher(**settings)
        result = stitcher.stitch(
            [
                "boat5.jpg",
                "s1.jpg",
                "s2.jpg",
                "boat2.jpg",
                "boat3.jpg",
                "boat4.jpg",
                "boat1.jpg",
                "boat6.jpg",
            ]
        )
        cv.imwrite("subset_low_res.jpg", result)

        max_image_shape_derivation = 100
        np.testing.assert_allclose(
            result.shape[:2], (705, 3374), atol=max_image_shape_derivation
        )

    def test_stitcher_budapest(self):
        settings = {
            "confidence_threshold": 0.3,
            "crop": False,
        }

        stitcher = AffineStitcher(**settings)
        result = stitcher.stitch(
            [
                "budapest1.jpg",
                "budapest2.jpg",
                "budapest3.jpg",
                "budapest4.jpg",
                "budapest5.jpg",
                "budapest6.jpg",
            ]
        )

        cv.imwrite("budapest.jpg", result)

        max_image_shape_derivation = 50
        np.testing.assert_allclose(
            result.shape[:2], (1155, 2310), atol=max_image_shape_derivation
        )


def starttest():
    unittest.main()


if __name__ == "__main__":
    starttest()
