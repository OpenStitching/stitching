import unittest

from .context import FeatureDetector, load_test_img


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


def start_test():
    unittest.main()


if __name__ == "__main__":
    start_test()
