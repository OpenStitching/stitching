import math

import cv2 as cv
import numpy as np


class FeatureMatcher:
    """https://docs.opencv.org/4.x/da/d87/classcv_1_1detail_1_1FeaturesMatcher.html"""

    MATCHER_CHOICES = ("homography", "affine")
    DEFAULT_MATCHER = "homography"
    DEFAULT_RANGE_WIDTH = -1

    def __init__(
        self, matcher_type=DEFAULT_MATCHER, range_width=DEFAULT_RANGE_WIDTH, **kwargs
    ):
        """
        初始化FeatureMatcher类
        :param matcher_type: 匹配器类型
        :param range_width: 范围宽度
        :param kwargs: 其他参数
        """
        if matcher_type == "affine":
            self.matcher = cv.detail_AffineBestOf2NearestMatcher(**kwargs)
        elif range_width == -1:
            self.matcher = cv.detail_BestOf2NearestMatcher(**kwargs)
        else:
            self.matcher = cv.detail_BestOf2NearestRangeMatcher(range_width, **kwargs)

    def match_features(self, features, *args, **kwargs):
        """
        匹配特征点
        :param features: 特征点
        :param args: 其他参数
        :param kwargs: 其他参数
        :return: 成对匹配
        """
        pairwise_matches = self.matcher.apply2(features, *args, **kwargs)
        self.matcher.collectGarbage()
        return pairwise_matches

    @staticmethod
    def draw_matches_matrix(
        imgs, features, matches, conf_thresh=1, inliers=False, **kwargs
    ):
        """
        绘制匹配矩阵
        :param imgs: 图像列表
        :param features: 特征点列表
        :param matches: 成对匹配
        :param conf_thresh: 置信度阈值
        :param inliers: 是否只绘制内点
        :param kwargs: 其他参数
        :return: 匹配矩阵图像生成器
        """
        matches_matrix = FeatureMatcher.get_matches_matrix(matches)
        for idx1, idx2 in FeatureMatcher.get_all_img_combinations(len(imgs)):
            match = matches_matrix[idx1, idx2]
            if match.confidence < conf_thresh or len(match.matches) == 0:
                continue
            if inliers:
                kwargs["matchesMask"] = match.getInliers()
            yield idx1, idx2, FeatureMatcher.draw_matches(
                imgs[idx1], features[idx1], imgs[idx2], features[idx2], match, **kwargs
            )

    @staticmethod
    def draw_matches(img1, features1, img2, features2, match1to2, **kwargs):
        """
        绘制匹配结果
        :param img1: 图像1
        :param features1: 图像1的特征点
        :param img2: 图像2
        :param features2: 图像2的特征点
        :param match1to2: 图像1到图像2的匹配结果
        :param kwargs: 其他参数
        :return: 绘制了匹配结果的图像
        """
        kwargs.setdefault("flags", cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        keypoints1 = features1.getKeypoints()
        keypoints2 = features2.getKeypoints()
        matches = match1to2.getMatches()

        return cv.drawMatches(
            img1, keypoints1, img2, keypoints2, matches, None, **kwargs
        )

    @staticmethod
    def get_matches_matrix(pairwise_matches):
        """
        获取匹配矩阵
        :param pairwise_matches: 成对匹配
        :return: 匹配矩阵
        """
        return FeatureMatcher.array_in_square_matrix(pairwise_matches)

    @staticmethod
    def get_confidence_matrix(pairwise_matches):
        """
        获取置信度矩阵
        :param pairwise_matches: 成对匹配
        :return: 置信度矩阵
        """
        matches_matrix = FeatureMatcher.get_matches_matrix(pairwise_matches)
        match_confs = [[m.confidence for m in row] for row in matches_matrix]
        match_conf_matrix = np.array(match_confs)
        return match_conf_matrix

    @staticmethod
    def array_in_square_matrix(array):
        """
        将数组转换为方阵
        :param array: 数组
        :return: 方阵
        """
        matrix_dimension = int(math.sqrt(len(array)))
        rows = []
        for i in range(0, len(array), matrix_dimension):
            rows.append(array[i : i + matrix_dimension])
        return np.array(rows)

    def get_all_img_combinations(number_imgs):
        """
        获取所有图像组合
        :param number_imgs: 图像数量
        :return: 图像组合生成器
        """
        ii, jj = np.triu_indices(number_imgs, k=1)
        for i, j in zip(ii, jj):
            yield i, j

    @staticmethod
    def get_match_conf(match_conf, feature_detector_type):
        """
        获取匹配置信度
        :param match_conf: 匹配置信度
        :param feature_detector_type: 特征检测器类型
        :return: 匹配置信度
        """
        if match_conf is None:
            match_conf = FeatureMatcher.get_default_match_conf(feature_detector_type)
        return match_conf

    @staticmethod
    def get_default_match_conf(feature_detector_type):
        """
        获取默认匹配置信度
        :param feature_detector_type: 特征检测器类型
        :return: 默认匹配置信度
        """
        if feature_detector_type == "orb":
            return 0.3
        return 0.65
