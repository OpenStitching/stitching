import unittest

import numpy as np
from context import Images, NamedImages, NumpyImages, load_test_img, test_input


class TestStitcher(unittest.TestCase):
    # TODO: DYI, test np arrays, Factory

    def test_numpy_image_input(self):
        images = NumpyImages([load_test_img("s1.jpg"), load_test_img("s2.jpg")])

        full_np_arrays = list(images)

        np.testing.assert_array_equal(images.sizes, [(1246, 700), (1385, 700)])

        low_sizes = images.get_scaled_img_sizes(Images.Resolution.LOW)
        np.testing.assert_array_equal(low_sizes, [(422, 237), (469, 237)])

        low_np_arrays = list(images.resize(Images.Resolution.LOW))

        ratio = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.LOW)
        self.assertEqual(ratio, 0.408248290463863)

    def test_named_image_input(self):
        images = NamedImages([test_input("s1.jpg"), test_input("s2.jpg")])

        full_np_arrays = list(images)

        np.testing.assert_array_equal(images.sizes, [(1246, 700), (1385, 700)])

        low_sizes = images.get_scaled_img_sizes(Images.Resolution.LOW)
        np.testing.assert_array_equal(low_sizes, [(422, 237), (469, 237)])

        low_np_arrays = list(images.resize(Images.Resolution.LOW))

        ratio = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.LOW)
        self.assertEqual(ratio, 0.408248290463863)

    def test_images(self):
        self.assertEqual(Images.Resolution.LOW.name, "LOW")
        self.assertEqual(Images.Resolution.LOW.value, 0.1)


def start_test():
    unittest.main()


if __name__ == "__main__":
    start_test()
