import os
import sys
import unittest
from unittest.mock import patch

import cv2 as cv
import numpy as np

from .context import OUT_DIR, create_parser, main, testimg


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.parser = create_parser()

    def test_parser(self):
        parsed = self.parser.parse_args(["img*.jpg", "--low_megapix", "1"])
        self.assertEqual(parsed.low_megapix, 1.0)

    def test_main(self):
        output = os.path.join(OUT_DIR, "weir_from_cli.jpg")
        testargs = [
            "stitch.py",
            testimg("weir_?.jpg"),
            "--final_megapix",
            "0.05",
            "--output",
            output,
        ]
        with patch.object(sys, "argv", testargs):
            main()

            img = cv.imread(output)
            max_image_shape_derivation = 10
            np.testing.assert_allclose(
                img.shape[:2], (150, 590), atol=max_image_shape_derivation
            )


def starttest():
    unittest.main()


if __name__ == "__main__":
    starttest()
