import os
import unittest
from datetime import datetime

import numpy as np

from .context import (
    VERBOSE_DIR,
    AffineStitcher,
    Stitcher,
    StitchingError,
    StitchingWarning,
    load_test_img,
    test_input,
    test_output,
    write_test_result,
)


class TestStitcher(unittest.TestCase):
    def test_stitcher_weir(self):
        stitcher = Stitcher()
        max_derivation = 30
        expected_shape = (673, 2636)

        # from image filenames
        imgs = [test_input("weir*.jpg")]
        name = "weir_from_filenames"

        self.stitch_test_with_warning(
            stitcher,
            imgs,
            expected_shape,
            max_derivation,
            name,
            StitchingWarning,
            "Not all images are included",
        )

        # from loaded numpy arrays
        imgs = [
            load_test_img("weir_1.jpg"),
            load_test_img("weir_2.jpg"),
            load_test_img("weir_3.jpg"),
            load_test_img("weir_noise.jpg"),
        ]
        name = "weir_from_numpy_images"

        self.stitch_test_with_warning(
            stitcher,
            imgs,
            expected_shape,
            max_derivation,
            name,
            StitchingWarning,
            "Not all images are included",
        )

    def test_stitcher_with_not_matching_images(self):
        stitcher = Stitcher()
        imgs = [test_input("s1.jpg"), test_input("boat1.jpg")]

        self.stitch_test_with_error(
            stitcher,
            imgs,
            (),
            0,
            "",
            StitchingError,
            "No match exceeds the given confidence threshold",
            verbose=False,
        )

    def test_stitcher_aquaduct(self):
        stitcher = Stitcher(nfeatures=250, crop=False)
        imgs = [test_input("s?.jpg")]
        max_derivation = 3
        expected_shape = (700, 1811)
        name = "s_result"

        self.stitch_test(stitcher, imgs, expected_shape, max_derivation, name)

    def test_stitcher_boat1(self):
        settings = {
            "warper_type": "fisheye",
            "wave_correct_kind": "no",
            "finder": "dp_colorgrad",
            "compensator": "no",
            "crop": False,
        }
        stitcher = Stitcher(**settings)
        imgs = [
            test_input("boat5.jpg"),
            test_input("boat2.jpg"),
            test_input("boat3.jpg"),
            test_input("boat4.jpg"),
            test_input("boat1.jpg"),
            test_input("boat6.jpg"),
        ]
        max_derivation = 600
        expected_shape = (14488, 7556)
        name = "boat_fisheye"

        self.stitch_test(
            stitcher, imgs, expected_shape, max_derivation, name, verbose=False
        )

    def test_stitcher_boat2(self):
        settings = {
            "warper_type": "compressedPlaneA2B1",
            "finder": "dp_colorgrad",
            "compensator": "channel_blocks",
            "crop": False,
        }
        stitcher = Stitcher(**settings)
        imgs = [
            test_input("boat5.jpg"),
            test_input("boat2.jpg"),
            test_input("boat3.jpg"),
            test_input("boat4.jpg"),
            test_input("boat1.jpg"),
            test_input("boat6.jpg"),
        ]
        max_derivation = 600
        expected_shape = (7400, 12340)
        name = "boat_fisheye"

        self.stitch_test(
            stitcher, imgs, expected_shape, max_derivation, name, verbose=False
        )

    def test_stitcher_boat_aquaduct_subset(self):
        graph = test_output("boat_subset_matches_graph.txt")
        settings = {"final_megapix": 1, "matches_graph_dot_file": graph}
        stitcher = Stitcher(**settings)
        imgs = [
            test_input("boat5.jpg"),
            test_input("s1.jpg"),
            test_input("s2.jpg"),
            test_input("boat2.jpg"),
            test_input("boat3.jpg"),
            test_input("boat4.jpg"),
            test_input("boat1.jpg"),
            test_input("boat6.jpg"),
        ]
        max_derivation = 100
        expected_shape = (705, 3374)
        name = "boat_subset_low_res"

        self.stitch_test_with_warning(
            stitcher,
            imgs,
            expected_shape,
            max_derivation,
            name,
            StitchingWarning,
            "Not all images are included",
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
        imgs = [test_input("budapest?.jpg")]
        max_derivation = 50
        expected_shape = (1155, 2310)
        name = "budapest"

        self.stitch_test(stitcher, imgs, expected_shape, max_derivation, name)

    def test_stitcher_feature_masks(self):
        stitcher = Stitcher(crop=False)

        # without masks
        imgs = [test_input("barcode1.png"), test_input("barcode2.png")]
        max_derivation = 25
        expected_shape = (905, 2124)
        name = "features_without_mask"

        self.stitch_test(stitcher, imgs, expected_shape, max_derivation, name)

        # with masks
        masks = [test_input("mask1.png"), test_input("mask2.png")]
        max_derivation = 15
        expected_shape = (716, 1852)
        name = "features_with_mask"

        self.stitch_test(
            stitcher, imgs, expected_shape, max_derivation, name, feature_masks=masks
        )

    def stitch_test(
        self,
        stitcher,
        imgs,
        expected_shape,
        max_derivation,
        name,
        feature_masks=[],
        verbose=True,
    ):
        result = stitcher.stitch(imgs, feature_masks)

        if verbose:
            verbose_dir_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + name
            verbose_dir = os.path.join(VERBOSE_DIR, verbose_dir_name)
            os.makedirs(verbose_dir)
            result_verbose = stitcher.stitch_verbose(imgs, feature_masks, verbose_dir)
            np.testing.assert_allclose(
                result.shape, result_verbose.shape, atol=max_derivation
            )

        np.testing.assert_allclose(
            result.shape[:2], expected_shape, atol=max_derivation
        )

        write_test_result(name + ".jpg", result)

    def stitch_test_with_warning(
        self,
        stitcher,
        imgs,
        expected_shape,
        max_derivation,
        name,
        expected_warning_type,
        expected_warning_message,
        feature_masks=[],
        verbose=True,
    ):
        with self.assertWarns(expected_warning_type) as cm:
            self.stitch_test(
                stitcher,
                imgs,
                expected_shape,
                max_derivation,
                name,
                feature_masks,
                verbose,
            )
        self.assertTrue(str(cm.warning).startswith(expected_warning_message))

    def stitch_test_with_error(
        self,
        stitcher,
        imgs,
        expected_shape,
        max_derivation,
        name,
        expected_error_type,
        expected_error_message,
        feature_masks=[],
        verbose=True,
    ):
        with self.assertRaises(expected_error_type) as cm:
            self.stitch_test(
                stitcher,
                imgs,
                expected_shape,
                max_derivation,
                name,
                feature_masks,
                verbose,
            )
        self.assertTrue(str(cm.exception).startswith(expected_error_message))

    def test_use_of_a_stitcher_for_multiple_image_sets(self):
        # the scale should not be fixed by the first run but set dynamically
        # based on every input image set.
        stitcher = Stitcher()
        _ = stitcher.stitch([test_input("s1.jpg"), test_input("s2.jpg")])
        self.assertEqual(round(stitcher.images._scalers["MEDIUM"].scale, 2), 0.83)
        _ = stitcher.stitch([test_input("boat1.jpg"), test_input("boat2.jpg")])
        self.assertEqual(round(stitcher.images._scalers["MEDIUM"].scale, 2), 0.24)


def start_test():
    unittest.main()


if __name__ == "__main__":
    start_test()
