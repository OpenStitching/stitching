import os
import unittest

import cv2 as cv
import numpy as np

from .context import FeatureDetector, FeatureMatcher, Stitcher

TEST_DIR = os.path.abspath(os.path.dirname(__file__))


class TestRangeMatcher(unittest.TestCase):
    def test_BestOf2NearestRangeMatcher(self):
        img1 = cv.imread(os.path.join(TEST_DIR, "testdata", "weir_1.jpg"))
        img2 = cv.imread(os.path.join(TEST_DIR, "testdata", "weir_2.jpg"))
        img3 = cv.imread(os.path.join(TEST_DIR, "testdata", "weir_3.jpg"))

        detector = FeatureDetector("orb")
        features = [detector.detect_features(img) for img in [img1, img2, img3]]

        matcher = FeatureMatcher(range_width=1)
        pairwise_matches = matcher.match_features(features)
        conf_matrix = FeatureMatcher.get_confidence_matrix(pairwise_matches)

        self.assertTrue(
            np.array_equal(
                conf_matrix > 0,
                np.array(
                    [[False, True, False], [True, False, True], [False, True, False]]
                ),
            )
        )

    def test_matches_graph_issue56(self):
        settings = {
            "range_width": 1,
            "confidence_threshold": 0,
            "matches_graph_dot_file": "range_width_matches_graph.txt",
        }
        stitcher = Stitcher(**settings)
        stitcher.stitch(
            [
                os.path.join(TEST_DIR, "testdata", "weir_1.jpg"),
                os.path.join(TEST_DIR, "testdata", "weir_2.jpg"),
                os.path.join(TEST_DIR, "testdata", "weir_3.jpg"),
            ]
        )

        # TODO: Automated test that matches graph is correct
