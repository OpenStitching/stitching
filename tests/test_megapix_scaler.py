import os
import unittest

import cv2 as cv

from .context import MegapixDownscaler, MegapixScaler


class TestScaler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEST_DIR = os.path.abspath(os.path.dirname(__file__))
        os.chdir(os.path.join(TEST_DIR, "testdata"))
        cls.img = cv.imread("s1.jpg")
        cls.size = (cls.img.shape[1], cls.img.shape[0])

    def test_get_scale_by_resolution(self):
        scaler = MegapixScaler(0.6)

        scale = scaler.get_scale_by_resolution(1_200_000)

        self.assertEqual(scale, 0.7071067811865476)

    def test_get_scale_by_image(self):
        scaler = MegapixScaler(0.6)

        scaler.set_scale_by_img_size(self.size)

        self.assertEqual(scaler.scale, 0.8294067854101966)

    def test_get_scaled_img_size(self):
        scaler = MegapixScaler(0.6)
        scaler.set_scale_by_img_size(self.size)

        size = scaler.get_scaled_img_size(self.size)
        self.assertEqual(size, (1033, 581))
        # 581*1033 = 600173 px = ~0.6 MP

    def test_force_of_downscaling(self):
        normal_scaler = MegapixScaler(2)
        downscaler = MegapixDownscaler(2)

        normal_scaler.set_scale_by_img_size(self.size)
        downscaler.set_scale_by_img_size(self.size)

        self.assertEqual(normal_scaler.scale, 1.5142826857233715)
        self.assertEqual(downscaler.scale, 1.0)


def starttest():
    unittest.main()


if __name__ == "__main__":
    starttest()
