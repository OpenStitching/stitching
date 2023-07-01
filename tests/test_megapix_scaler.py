import unittest

from .context import MegapixDownscaler, MegapixScaler

SIZE = (1246, 700)


class TestScaler(unittest.TestCase):
    def test_get_scale_by_resolution(self):
        scaler = MegapixScaler(0.6)

        scale = scaler.get_scale_by_resolution(1_200_000)

        self.assertEqual(scale, 0.7071067811865476)

    def test_get_scale_by_image(self):
        scaler = MegapixScaler(0.6)

        scaler.set_scale_by_img_size(SIZE)

        self.assertEqual(scaler.scale, 0.8294067854101966)

    def test_get_scaled_img_size(self):
        scaler = MegapixScaler(0.6)
        scaler.set_scale_by_img_size(SIZE)

        size = scaler.get_scaled_img_size(SIZE)
        self.assertEqual(size, (1033, 581))
        # 581*1033 = 600173 px = ~0.6 MP

    def test_force_of_downscaling(self):
        normal_scaler = MegapixScaler(2)
        downscaler = MegapixDownscaler(2)

        normal_scaler.set_scale_by_img_size(SIZE)
        downscaler.set_scale_by_img_size(SIZE)

        self.assertEqual(normal_scaler.scale, 1.5142826857233715)
        self.assertEqual(downscaler.scale, 1.0)


def start_test():
    unittest.main()


if __name__ == "__main__":
    start_test()
