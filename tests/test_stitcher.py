import os
import unittest

import cv2 as cv
import numpy as np
from context import AffineStitcher, Stitcher, testimg, testresult


class TestStitcher(unittest.TestCase):
    def test_stitcher_aquaduct(self):
        stitcher = Stitcher(nfeatures=250, crop=False)
        result = stitcher.stitch([testimg("s?.jpg")])
        testresult("s_result.jpg", result)

        max_image_shape_derivation = 3
        np.testing.assert_allclose(
            result.shape[:2], (700, 1811), atol=max_image_shape_derivation
        )

    def test_stitcher_boat1(self):
        settings = {
            "warper_type": "fisheye",
            "wave_correct_kind": "no",
            "finder": "dp_colorgrad",
            "compensator": "no",
            "crop": False,
        }

        stitcher = Stitcher(**settings)
        result = stitcher.stitch(
            [
                testimg("boat5.jpg"),
                testimg("boat2.jpg"),
                testimg("boat3.jpg"),
                testimg("boat4.jpg"),
                testimg("boat1.jpg"),
                testimg("boat6.jpg"),
            ]
        )

        testresult("boat_fisheye.jpg", result)

        max_image_shape_derivation = 600
        np.testing.assert_allclose(
            result.shape[:2], (14488, 7556), atol=max_image_shape_derivation
        )

    def test_stitcher_boat2(self):
        settings = {
            "warper_type": "compressedPlaneA2B1",
            "finder": "dp_colorgrad",
            "compensator": "channel_blocks",
            "crop": False,
        }

        stitcher = Stitcher(**settings)
        result = stitcher.stitch(
            [
                testimg("boat5.jpg"),
                testimg("boat2.jpg"),
                testimg("boat3.jpg"),
                testimg("boat4.jpg"),
                testimg("boat1.jpg"),
                testimg("boat6.jpg"),
            ]
        )

        testresult("boat_plane.jpg", result)

        max_image_shape_derivation = 600
        np.testing.assert_allclose(
            result.shape[:2], (7400, 12340), atol=max_image_shape_derivation
        )

    def test_stitcher_boat_aquaduct_subset(self):
        settings = {"final_megapix": 1}

        stitcher = Stitcher(**settings)
        result = stitcher.stitch(
            [
                testimg("boat5.jpg"),
                testimg("s1.jpg"),
                testimg("s2.jpg"),
                testimg("boat2.jpg"),
                testimg("boat3.jpg"),
                testimg("boat4.jpg"),
                testimg("boat1.jpg"),
                testimg("boat6.jpg"),
            ]
        )
        testresult("subset_low_res.jpg", result)

        max_image_shape_derivation = 100
        np.testing.assert_allclose(
            result.shape[:2], (705, 3374), atol=max_image_shape_derivation
        )

    def test_affine_stitcher_budapest(self):
        settings = {
            "detector": "sift",
            "crop": False,
        }

        stitcher = AffineStitcher(**settings)
        result = stitcher.stitch([testimg("budapest?.jpg")])

        testresult("budapest.jpg", result)

        max_image_shape_derivation = 50
        np.testing.assert_allclose(
            result.shape[:2], (1155, 2310), atol=max_image_shape_derivation
        )


def starttest():
    unittest.main()


if __name__ == "__main__":
    starttest()
