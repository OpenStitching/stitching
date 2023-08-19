import unittest

import numpy as np

from .context import FeatureDetector, StitchingError, load_test_img


class TestFeatureDetector(unittest.TestCase):
    def test_number_of_keypoints(self):
        img1 = load_test_img("s1.jpg")

        default_number_of_keypoints = 500
        detector = FeatureDetector("orb")
        features = detector.detect_features(img1)
        self.assertEqual(len(features.getKeypoints()), default_number_of_keypoints)

        other_keypoints = 1000
        detector = FeatureDetector("orb", nfeatures=other_keypoints)
        features = detector.detect_features(img1)
        self.assertEqual(len(features.getKeypoints()), other_keypoints)

    def test_feature_masking(self):
        img1 = load_test_img("s1.jpg")

        # creating the image mask and setting only the middle 20% as enabled
        height, width = img1.shape[:2]
        top, bottom, left, right = map(
            int, (0.4 * height, 0.6 * height, 0.4 * width, 0.6 * width)
        )
        mask = np.zeros(shape=(height, width), dtype=np.uint8)
        mask[top:bottom, left:right] = 255

        num_features = 1000
        detector = FeatureDetector("orb", nfeatures=num_features)
        keypoints = detector.detect_features(img1, mask=mask).getKeypoints()
        self.assertTrue(len(keypoints) > 0)
        for point in keypoints:
            x, y = point.pt
            self.assertTrue(left <= x < right)
            self.assertTrue(top <= y < bottom)

    def test_feature_mask_validation(self):
        img1 = load_test_img("barcode1.png")
        img2 = load_test_img("barcode2.png")
        mask1 = load_test_img("mask1.png", 0)
        mask2 = load_test_img("mask2.png", 0)

        detector = FeatureDetector()
        with self.assertRaises(StitchingError) as cm:
            detector.detect_with_masks([img1, img2], [mask2, mask1])
        self.assertTrue(str(cm.exception).startswith("Resolution of mask 1"))

        with self.assertRaises(StitchingError) as cm:
            detector.detect_with_masks([img1, img2], [mask1])
        self.assertTrue(str(cm.exception).startswith("image and mask lists"))

        features = detector.detect_with_masks([img1, img2], [mask1, mask2])
        self.assertEqual(len(features), 2)


def start_test():
    unittest.main()


if __name__ == "__main__":
    start_test()
