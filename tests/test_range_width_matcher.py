import unittest

import numpy as np

from .context import (
    FeatureDetector,
    FeatureMatcher,
    Stitcher,
    load_test_img,
    test_input,
    test_output,
)


class TestRangeMatcher(unittest.TestCase):
    def test_BestOf2NearestRangeMatcher(self):
        img1 = load_test_img("weir_1.jpg")
        img2 = load_test_img("weir_2.jpg")
        img3 = load_test_img("weir_3.jpg")

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
            "matches_graph_dot_file": test_output("range_width_matches_graph.txt"),
        }
        stitcher = Stitcher(**settings)
        stitcher.stitch(
            [
                test_input("weir_1.jpg"),
                test_input("weir_2.jpg"),
                test_input("weir_3.jpg"),
            ]
        )

        # TODO: Automated test that matches graph is correct
