import numpy as np


class MegapixScaler:
    def __init__(self, megapix):
        """
        初始化MegapixScaler类
        :param megapix: 目标百万像素数
        """
        self.megapix = megapix
        self.is_scale_set = False
        self.scale = None

    def set_scale_by_img_size(self, img_size):
        """
        根据图像尺寸设置缩放比例
        :param img_size: 图像尺寸
        """
        self.set_scale(self.get_scale_by_resolution(img_size[0] * img_size[1]))

    def set_scale(self, scale):
        """
        设置缩放比例
        :param scale: 缩放比例
        """
        self.scale = scale
        self.is_scale_set = True

    def get_scale_by_resolution(self, resolution):
        """
        根据分辨率获取缩放比例
        :param resolution: 分辨率
        :return: 缩放比例
        """
        if self.megapix > 0:
            return np.sqrt(self.megapix * 1e6 / resolution)
        return 1.0

    def get_scaled_img_size(self, img_size):
        """
        获取缩放后的图像尺寸
        :param img_size: 原始图像尺寸
        :return: 缩放后的图像尺寸
        """
        width = int(round(img_size[0] * self.scale))
        height = int(round(img_size[1] * self.scale))
        return (width, height)


class MegapixDownscaler(MegapixScaler):
    @staticmethod
    def force_downscale(scale):
        """
        强制缩小比例
        :param scale: 缩放比例
        :return: 缩小后的比例
        """
        return min(1.0, scale)

    def set_scale(self, scale):
        """
        设置缩小后的比例
        :param scale: 缩放比例
        """
        scale = self.force_downscale(scale)
        super().set_scale(scale)
