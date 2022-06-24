from collections import OrderedDict

import cv2 as cv


class FeatureDetector:
    """https://docs.opencv.org/4.x/d0/d13/classcv_1_1Feature2D.html"""

    DETECTOR_CHOICES = OrderedDict()

    DETECTOR_CHOICES["orb"] = cv.ORB.create
    DETECTOR_CHOICES["sift"] = cv.SIFT_create
    DETECTOR_CHOICES["brisk"] = cv.BRISK_create
    DETECTOR_CHOICES["akaze"] = cv.AKAZE_create

    DEFAULT_DETECTOR = list(DETECTOR_CHOICES.keys())[0]

    def __init__(self, detector=DEFAULT_DETECTOR, **kwargs):
        self.detector = FeatureDetector.DETECTOR_CHOICES[detector](**kwargs)

    def detect_features(self, img, *args, **kwargs):
        return cv.detail.computeImageFeatures2(self.detector, img, *args, **kwargs)

    @staticmethod
    def draw_keypoints(img, features, **kwargs):
        kwargs.setdefault("color", (0, 255, 0))
        keypoints = features.getKeypoints()
        return cv.drawKeypoints(img, keypoints, None, **kwargs)
