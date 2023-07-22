import unittest
import numpy as np

from .context import InputImages, test_input

class TestStitcher(unittest.TestCase):

    def test_input_images(self):
        input_images = InputImages([test_input("s?.jpg")])
        input_images.set_resolutions()
        imgs = list(input_images.read_and_resize(input_images.resolutions.MEDIUM))
        
        np.testing.assert_array_equal(input_images.img_sizes, [(1246, 700), (1385, 700)])
        
        low_sizes = input_images.get_scaled_img_sizes(input_images.resolutions.LOW)
        np.testing.assert_array_equal(low_sizes, [(422, 237), (469, 237)])
        
        ratio = input_images.get_ratio(input_images.resolutions.MEDIUM, input_images.resolutions.LOW)
        self.assertEqual(ratio, 0.408248290463863)

def start_test():
    unittest.main()


if __name__ == "__main__":
    start_test()


