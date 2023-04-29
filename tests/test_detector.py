import unittest

import cv2 as cv
import numpy as np

from .context import FeatureDetector, FeatureMatcher, Subsetter, testimg


class TestFeatureDetector(unittest.TestCase):

    def test_number_of_keypoints(self):
        img1 = cv.imread(testimg("s1.jpg"))

        default_number_of_keypoints = 500
        detector = FeatureDetector("orb")
        features = detector.detect_features(img1)
        self.assertEqual(len(features.getKeypoints()), default_number_of_keypoints)

        other_keypoints = 1000
        detector = FeatureDetector("orb", nfeatures=other_keypoints)
        features = detector.detect_features(img1)
        self.assertEqual(len(features.getKeypoints()), other_keypoints)


def starttest():
    unittest.main()


if __name__ == "__main__":
    starttest()
