from collections import OrderedDict

import cv2 as cv
import numpy as np

from .stitching_error import StitchingError


class FeatureDetector:
    """https://docs.opencv.org/4.x/d0/d13/classcv_1_1Feature2D.html"""

    DETECTOR_CHOICES = OrderedDict()

    DETECTOR_CHOICES["orb"] = cv.ORB.create
    DETECTOR_CHOICES["sift"] = cv.SIFT_create
    DETECTOR_CHOICES["brisk"] = cv.BRISK_create
    DETECTOR_CHOICES["akaze"] = cv.AKAZE_create

    DEFAULT_DETECTOR = list(DETECTOR_CHOICES.keys())[0]

    def __init__(self, detector=DEFAULT_DETECTOR, **kwargs):
        """
        初始化FeatureDetector类
        :param detector: 特征检测器类型
        :param kwargs: 其他参数
        """
        self.detector = FeatureDetector.DETECTOR_CHOICES[detector](**kwargs)

    def detect_features(self, img, *args, **kwargs):
        """
        检测图像中的特征
        :param img: 图像
        :param args: 其他参数
        :param kwargs: 其他参数
        :return: 检测到的特征
        """
        return cv.detail.computeImageFeatures2(self.detector, img, *args, **kwargs)

    def detect(self, imgs):
        """
        检测图像列表中的特征
        :param imgs: 图像列表
        :return: 检测到的特征列表
        """
        return [self.detect_features(img) for img in imgs]

    def detect_with_masks(self, imgs, masks):
        """
        使用掩码检测图像列表中的特征
        :param imgs: 图像列表
        :param masks: 掩码列表
        :return: 检测到的特征列表
        """
        features = []
        for idx, (img, mask) in enumerate(zip(imgs, masks)):
            assert len(img.shape) == 3 and len(mask.shape) == 2
            if not len(imgs) == len(masks):
                raise StitchingError("image and mask lists must be of same length")
            if not np.array_equal(img.shape[:2], mask.shape):
                raise StitchingError(
                    f"Resolution of mask {idx + 1} {mask.shape} does not match"
                    f" the resolution of image {idx + 1} {img.shape[:2]}."
                )
            features.append(self.detect_features(img, mask=mask))
        return features

    @staticmethod
    def draw_keypoints(img, features, **kwargs):
        """
        在图像上绘制特征点
        :param img: 图像
        :param features: 特征点
        :param kwargs: 其他参数
        :return: 绘制了特征点的图像
        """
        kwargs.setdefault("color", (0, 255, 0))
        keypoints = features.getKeypoints()
        return cv.drawKeypoints(img, keypoints, None, **kwargs)
