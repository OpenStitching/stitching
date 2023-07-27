import unittest

import numpy as np

from .context import Images, test_input


class TestStitcher(unittest.TestCase):
    def test_images(self):
        images = Images([test_input("s1.jpg"), test_input("s2.jpg")])
        _ = list(images.resize(images.read(), Images.Resolution.MEDIUM))

        np.testing.assert_array_equal(images.img_sizes, [(1246, 700), (1385, 700)])

        low_sizes = images.get_scaled_img_sizes(Images.Resolution.LOW)
        np.testing.assert_array_equal(low_sizes, [(422, 237), (469, 237)])

        ratio = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.LOW)
        self.assertEqual(ratio, 0.408248290463863)

        self.assertEqual(Images.Resolution.LOW.name, "LOW")
        self.assertEqual(Images.Resolution.LOW.value, 0.1)


def start_test():
    unittest.main()


if __name__ == "__main__":
    start_test()
