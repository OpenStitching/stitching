from itertools import chain

import cv2 as cv
import numpy as np

from .feature_matcher import FeatureMatcher
from .stitching_error import StitchingError


class Subsetter:
    """https://docs.opencv.org/4.x/d7/d74/group__stitching__rotation.html#ga855d2fccbcfc3b3477b34d415be5e786 and
    https://docs.opencv.org/4.x/d7/d74/group__stitching__rotation.html#gabaeb9dab170ea8066ae2583bf3a669e9"""  # noqa

    DEFAULT_CONFIDENCE_THRESHOLD = 1
    DEFAULT_MATCHES_GRAPH_DOT_FILE = None

    def __init__(
        self,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        matches_graph_dot_file=DEFAULT_MATCHES_GRAPH_DOT_FILE,
    ):
        self.confidence_threshold = confidence_threshold
        self.save_file = matches_graph_dot_file

    def subset(self, img_names, img_sizes, imgs, features, matches):
        self.save_matches_graph_dot_file(img_names, matches)
        indices = self.get_indices_to_keep(features, matches)

        img_names = Subsetter.subset_list(img_names, indices)
        img_sizes = Subsetter.subset_list(img_sizes, indices)
        imgs = Subsetter.subset_list(imgs, indices)
        features = Subsetter.subset_list(features, indices)
        matches = Subsetter.subset_matches(matches, indices)
        return img_names, img_sizes, imgs, features, matches

    def save_matches_graph_dot_file(self, img_names, pairwise_matches):
        if self.save_file:
            with open(self.save_file, "w") as filehandler:
                filehandler.write(self.get_matches_graph(img_names, pairwise_matches))

    def get_matches_graph(self, img_names, pairwise_matches):
        return cv.detail.matchesGraphAsString(
            img_names,
            pairwise_matches,
            0.00001  # see issue #56
            if (self.confidence_threshold == 0)
            else self.confidence_threshold,
        )

    def get_indices_to_keep(self, features, pairwise_matches):
        indices = cv.detail.leaveBiggestComponent(
            features, pairwise_matches, self.confidence_threshold
        )

        if len(indices) < 2:
            raise StitchingError("No match exceeds the " "given confidence threshold.")

        return indices

    @staticmethod
    def subset_list(list_to_subset, indices):
        return [list_to_subset[i] for i in indices]

    @staticmethod
    def subset_matches(pairwise_matches, indices):
        matches_matrix = FeatureMatcher.get_matches_matrix(pairwise_matches)
        matches_matrix_subset = matches_matrix[np.ix_(indices, indices)]
        matches_subset_list = list(chain.from_iterable(matches_matrix_subset.tolist()))
        return matches_subset_list
