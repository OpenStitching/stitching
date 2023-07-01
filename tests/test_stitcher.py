import unittest

import numpy as np

from .context import (
    AffineStitcher,
    Stitcher,
    StitchingError,
    StitchingWarning,
    test_input,
    test_output,
    write_test_result,
)


class TestStitcher(unittest.TestCase):
    def test_stitcher_with_not_matching_images(self):
        stitcher = Stitcher()
        with self.assertRaises(StitchingError) as cm:
            stitcher.stitch([test_input("s1.jpg"), test_input("boat1.jpg")])
        self.assertTrue(
            "No match exceeds the given confidence threshold" in str(cm.exception)
        )

    def test_stitcher_aquaduct(self):
        stitcher = Stitcher(nfeatures=250, crop=False)
        result = stitcher.stitch([test_input("s?.jpg")])
        write_test_result("s_result.jpg", result)

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
                test_input("boat5.jpg"),
                test_input("boat2.jpg"),
                test_input("boat3.jpg"),
                test_input("boat4.jpg"),
                test_input("boat1.jpg"),
                test_input("boat6.jpg"),
            ]
        )

        write_test_result("boat_fisheye.jpg", result)

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
                test_input("boat5.jpg"),
                test_input("boat2.jpg"),
                test_input("boat3.jpg"),
                test_input("boat4.jpg"),
                test_input("boat1.jpg"),
                test_input("boat6.jpg"),
            ]
        )

        write_test_result("boat_plane.jpg", result)

        max_image_shape_derivation = 600
        np.testing.assert_allclose(
            result.shape[:2], (7400, 12340), atol=max_image_shape_derivation
        )

    def test_stitcher_boat_aquaduct_subset(self):
        graph = test_output("boat_subset_matches_graph.txt")
        settings = {"final_megapix": 1, "matches_graph_dot_file": graph}

        stitcher = Stitcher(**settings)

        with self.assertWarns(StitchingWarning) as cm:
            result = stitcher.stitch(
                [
                    test_input("boat5.jpg"),
                    test_input("s1.jpg"),
                    test_input("s2.jpg"),
                    test_input("boat2.jpg"),
                    test_input("boat3.jpg"),
                    test_input("boat4.jpg"),
                    test_input("boat1.jpg"),
                    test_input("boat6.jpg"),
                ]
            )

        self.assertTrue(str(cm.warning).startswith("Not all images are included"))

        write_test_result("boat_subset_low_res.jpg", result)

        max_image_shape_derivation = 100
        np.testing.assert_allclose(
            result.shape[:2], (705, 3374), atol=max_image_shape_derivation
        )

        with open(graph, "r") as file:
            graph_content = file.read()
            self.assertTrue(graph_content.startswith("graph matches_graph{"))

    def test_affine_stitcher_budapest(self):
        settings = {
            "detector": "sift",
            "crop": False,
        }

        stitcher = AffineStitcher(**settings)
        result = stitcher.stitch([test_input("budapest?.jpg")])

        write_test_result("budapest.jpg", result)

        max_image_shape_derivation = 50
        np.testing.assert_allclose(
            result.shape[:2], (1155, 2310), atol=max_image_shape_derivation
        )


def start_test():
    unittest.main()


if __name__ == "__main__":
    start_test()
