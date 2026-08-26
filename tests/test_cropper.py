import unittest
from unittest.mock import patch

import numpy as np

from .context import Cropper, Rectangle


class TestCropper(unittest.TestCase):
    def test_prepare_ignores_rectangles_outside_lir(self):
        cropper = Cropper()
        imgs = [np.full((10, 10), idx, dtype=np.uint8) for idx in range(3)]
        masks = [np.ones((10, 10), dtype=np.uint8) for _ in imgs]
        corners = [(0, 0), (20, 20), (5, 5)]
        sizes = [(10, 10)] * len(imgs)
        lir = Rectangle(2, 2, 12, 12)

        with patch.object(cropper, "estimate_panorama_mask"), patch.object(
            cropper, "estimate_largest_interior_rectangle", return_value=lir
        ):
            cropper.prepare(imgs, masks, corners, sizes)

        cropped_imgs = list(cropper.crop_images(iter(imgs)))
        cropped_corners, cropped_sizes = cropper.crop_rois(corners, sizes)

        self.assertEqual(cropper.overlapping_indices, [0, 2])
        self.assertEqual(len(cropped_imgs), 2)
        self.assertEqual(cropped_imgs[0].shape, (8, 8))
        self.assertEqual(cropped_imgs[1].shape, (9, 9))
        self.assertTrue(np.all(cropped_imgs[0] == 0))
        self.assertTrue(np.all(cropped_imgs[1] == 2))
        self.assertEqual(cropped_corners, [(0, 0), (3, 3)])
        self.assertEqual(cropped_sizes, [(8, 8), (9, 9)])


if __name__ == "__main__":
    unittest.main()
