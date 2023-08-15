import unittest

import numpy as np

from .context import Images, _FilenameImages, _NumpyImages, load_test_img, test_input


class TestImages(unittest.TestCase):
    def test_numpy_image_input(self):
        images = Images.of([load_test_img("s1.jpg"), load_test_img("s2.jpg")])
        self.assertTrue(isinstance(images, _NumpyImages))
        self.assertEqual(images.names, ["1", "2"])
        self.check_s_images(images)

    def test_named_image_input(self):
        images = Images.of([test_input("s1.jpg"), test_input("s2.jpg")])
        self.assertTrue(isinstance(images, _FilenameImages))
        self.assertTrue(images.names[0].endswith("s1.jpg"))
        self.assertTrue(images.names[1].endswith("s2.jpg"))
        self.check_s_images(images)

    def check_s_images(self, images):
        self.assertTrue(isinstance(images, Images))

        full_np_arrays = list(images)
        shapes = [img.shape for img in full_np_arrays]
        np.testing.assert_array_equal(shapes, [(700, 1246, 3), (700, 1385, 3)])

        np.testing.assert_array_equal(images.sizes, [(1246, 700), (1385, 700)])

        low_sizes = images.get_scaled_img_sizes(Images.Resolution.LOW)
        np.testing.assert_array_equal(low_sizes, [(422, 237), (469, 237)])

        low_np_arrays1 = list(images.resize(Images.Resolution.LOW))
        low_np_arrays2 = list(images.resize(Images.Resolution.LOW, full_np_arrays))
        low_shapes1 = [img.shape for img in low_np_arrays1]
        low_shapes2 = [img.shape for img in low_np_arrays2]
        np.testing.assert_array_equal(low_shapes1, low_shapes2)
        np.testing.assert_array_equal(low_shapes1, (((237, 422, 3), (237, 469, 3))))

        ratio = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.LOW)
        self.assertEqual(ratio, 0.408248290463863)

    def test_images(self):
        self.assertEqual(Images.Resolution.LOW.name, "LOW")
        self.assertEqual(Images.Resolution.LOW.value, 0.1)

        images = Images.of(["1", "2"], 10)
        self.assertEqual(images._scalers["MEDIUM"].megapix, 10)


def start_test():
    unittest.main()


if __name__ == "__main__":
    start_test()
