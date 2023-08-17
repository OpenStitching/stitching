import os
import sys
import unittest
from datetime import datetime
from unittest.mock import patch

import cv2 as cv
import numpy as np

from .context import create_parser, main, test_input, test_output


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.parser = create_parser()

    def test_parser(self):
        parsed = self.parser.parse_args(["img*.jpg", "--low_megapix", "1"])
        self.assertEqual(parsed.low_megapix, 1.0)

    def test_main(self):
        output = test_output("weir_from_cli.jpg")
        test_args = [
            "stitch.py",
            test_input("weir_?.jpg"),
            "--final_megapix",
            "0.05",
            "--output",
            output,
        ]
        with patch.object(sys, "argv", test_args):
            main()

            img = cv.imread(output)
            max_image_shape_derivation = 10
            np.testing.assert_allclose(
                img.shape[:2], (150, 590), atol=max_image_shape_derivation
            )

    def test_main_verbose(self):
        name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_verbose_results"
        output = test_output(name)
        test_args = [
            "stitch.py",
            test_input("weir_?.jpg"),
            "--final_megapix",
            "0.05",
            "--verbose",
            "--verbose_dir",
            output,
        ]
        with patch.object(sys, "argv", test_args):
            main()

            img = cv.imread(os.path.join(output, "09_result.jpg"))
            max_image_shape_derivation = 10
            np.testing.assert_allclose(
                img.shape[:2], (150, 590), atol=max_image_shape_derivation
            )

    def test_main_feature_masks(self):
        output = test_output("features_with_mask_from_cli.jpg")
        test_args = [
            "stitch.py",
            test_input("barcode1.png"),
            test_input("barcode2.png"),
            "--feature_masks",
            test_input("mask1.png"),
            test_input("mask2.png"),
            "--output",
            output,
        ]
        with patch.object(sys, "argv", test_args):
            main()

            img = cv.imread(output)
            max_image_shape_derivation = 15
            np.testing.assert_allclose(
                img.shape[:2], (716, 1852), atol=max_image_shape_derivation
            )


def start_test():
    unittest.main()


if __name__ == "__main__":
    start_test()
