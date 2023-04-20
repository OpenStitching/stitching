import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import cv2 as cv

from .context import create_parser, testimg, OUT_DIR, main


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.parser = create_parser()

    def test_parser(self):
        parsed = self.parser.parse_args(["img*.jpg", "--low_megapix", "1"])
        self.assertEqual(parsed.low_megapix, 1.0)
        
    def test_main(self):
        output = os.path.join(OUT_DIR, "weir_from_cli.jpg")
        testargs = ["stitch.py", testimg("weir_?.jpg"), "--final_megapix", "0.05", "--output", output]
        with patch.object(sys, 'argv', testargs):
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
