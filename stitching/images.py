import os
from abc import ABC, abstractmethod
from enum import Enum
from glob import glob

import cv2 as cv
import numpy as np

from .megapix_scaler import MegapixDownscaler
from .stitching_error import StitchingError


class Images(ABC):
    class Resolution(Enum):
        MEDIUM = 0.6
        LOW = 0.1
        FINAL = -1

    @staticmethod
    def of(
        images,
        medium_megapix=Resolution.MEDIUM.value,
        low_megapix=Resolution.LOW.value,
        final_megapix=Resolution.FINAL.value,
    ):
        """
        创建Images对象
        :param images: 图像列表
        :param medium_megapix: 中等分辨率
        :param low_megapix: 低分辨率
        :param final_megapix: 最终分辨率
        :return: Images对象
        """
        if not isinstance(images, list):
            raise StitchingError("images must be a list of images or filenames")
        if len(images) == 0:
            raise StitchingError("images must not be an empty list")

        if Images.check_list_element_types(images, np.ndarray):
            return _NumpyImages(images, medium_megapix, low_megapix, final_megapix)
        elif Images.check_list_element_types(images, str):
            return _FilenameImages(images, medium_megapix, low_megapix, final_megapix)
        else:
            raise StitchingError(
                """invalid images list:
                    must be numpy arrays (loaded images) or filename strings"""
            )

    @abstractmethod
    def __init__(self, images, medium_megapix, low_megapix, final_megapix):
        """
        初始化Images类
        :param images: 图像列表
        :param medium_megapix: 中等分辨率
        :param low_megapix: 低分辨率
        :param final_megapix: 最终分辨率
        """
        if medium_megapix < low_megapix:
            raise StitchingError(
                "Medium resolution megapix need to be "
                "greater or equal than low resolution "
                "megapix"
            )

        self._scalers = {}
        self._scalers["MEDIUM"] = MegapixDownscaler(medium_megapix)
        self._scalers["LOW"] = MegapixDownscaler(low_megapix)
        self._scalers["FINAL"] = MegapixDownscaler(final_megapix)
        self._scales_set = False

        self._sizes_set = False
        self._names_set = False

    @property
    def sizes(self):
        """
        获取图像尺寸列表
        :return: 图像尺寸列表
        """
        assert self._sizes_set
        return self._sizes

    @property
    def names(self):
        """
        获取图像名称列表
        :return: 图像名称列表
        """
        assert self._names_set
        return self._names

    @abstractmethod
    def subset(self, indices):
        """
        获取子集
        :param indices: 索引列表
        """
        self._sizes = [self._sizes[i] for i in indices]
        self._names = [self._names[i] for i in indices]

    def resize(self, resolution, imgs=None):
        """
        调整图像大小
        :param resolution: 分辨率
        :param imgs: 图像列表
        :return: 调整大小后的图像生成器
        """
        img_iterable = self.__iter__() if imgs is None else imgs
        for idx, img in enumerate(img_iterable):
            yield Images.resize_img_by_scaler(
                self._get_scaler(resolution), self._sizes[idx], img
            )

    @abstractmethod
    def __iter__(self):
        pass

    def _set_scales(self, size):
        """
        设置缩放比例
        :param size: 图像尺寸
        """
        if not self._scales_set:
            for scaler in self._scalers.values():
                scaler.set_scale_by_img_size(size)
            self._scales_set = True

    def _get_scaler(self, resolution):
        """
        获取缩放器
        :param resolution: 分辨率
        :return: 缩放器
        """
        Images.check_resolution(resolution)
        return self._scalers[resolution.name]

    def get_ratio(self, from_resolution, to_resolution):
        """
        获取分辨率比例
        :param from_resolution: 源分辨率
        :param to_resolution: 目标分辨率
        :return: 分辨率比例
        """
        assert self._scales_set
        Images.check_resolution(from_resolution)
        Images.check_resolution(to_resolution)
        return (
            self._get_scaler(to_resolution).scale
            / self._get_scaler(from_resolution).scale  # noqa: W503
        )

    def get_scaled_img_sizes(self, resolution):
        """
        获取缩放后的图像尺寸列表
        :param resolution: 分辨率
        :return: 缩放后的图像尺寸列表
        """
        assert self._scales_set and self._sizes_set
        Images.check_resolution(resolution)
        return [
            self._get_scaler(resolution).get_scaled_img_size(sz) for sz in self._sizes
        ]

    @staticmethod
    def read_image(img_name):
        """
        读取图像
        :param img_name: 图像文件名
        :return: 图像
        """
        img = cv.imread(img_name)
        if img is None:
            raise StitchingError("Cannot read image " + img_name)
        return img

    @staticmethod
    def get_image_size(img):
        """
        获取图像尺寸
        :param img: 图像
        :return: 图像尺寸（宽度，高度）
        """
        return (img.shape[1], img.shape[0])

    @staticmethod
    def resize_img_by_scaler(scaler, size, img):
        """
        使用缩放器调整图像大小
        :param scaler: 缩放器
        :param size: 图像尺寸
        :param img: 图像
        :return: 调整大小后的图像
        """
        desired_size = scaler.get_scaled_img_size(size)
        return cv.resize(img, desired_size, interpolation=cv.INTER_LINEAR_EXACT)

    @staticmethod
    def check_resolution(resolution):
        """
        检查分辨率
        :param resolution: 分辨率
        """
        assert isinstance(resolution, Enum) and resolution in Images.Resolution

    @staticmethod
    def resolve_wildcards(img_names):
        """
        解析通配符
        :param img_names: 图像文件名列表
        :return: 解析后的图像文件名列表
        """
        if len(img_names) == 1:
            img_names = [i for i in glob(img_names[0]) if not os.path.isdir(i)]
        return img_names

    @staticmethod
    def check_list_element_types(list_, type_):
        """
        检查列表元素类型
        :param list_: 列表
        :param type_: 类型
        :return: 是否全部为指定类型
        """
        return all([isinstance(element, type_) for element in list_])

    @staticmethod
    def to_binary(img):
        """
        转换为二值图像
        :param img: 图像
        :return: 二值图像
        """
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(img, 0.5, 255.0, cv.THRESH_BINARY)
        return binary.astype(np.uint8)


class _NumpyImages(Images):
    def __init__(self, images, medium_megapix, low_megapix, final_megapix):
        """
        初始化_NumpyImages类
        :param images: 图像列表
        :param medium_megapix: 中等分辨率
        :param low_megapix: 低分辨率
        :param final_megapix: 最终分辨率
        """
        super().__init__(images, medium_megapix, low_megapix, final_megapix)
        if len(images) < 2:
            raise StitchingError("2 or more Images needed")
        self._images = images
        self._sizes = [Images.get_image_size(img) for img in images]
        self._sizes_set = True
        self._names = [str(i + 1) for i in range(len(images))]
        self._names_set = True
        self._set_scales(self._sizes[0])

    def subset(self, indices):
        """
        获取子集
        :param indices: 索引列表
        """
        super().subset(indices)
        self._images = [self._images[i] for i in indices]

    def __iter__(self):
        """
        迭代图像
        :return: 图像生成器
        """
        for img in self._images:
            yield img


class _FilenameImages(Images):
    def __init__(self, images, medium_megapix, low_megapix, final_megapix):
        """
        初始化_FilenameImages类
        :param images: 图像文件名列表
        :param medium_megapix: 中等分辨率
        :param low_megapix: 低分辨率
        :param final_megapix: 最终分辨率
        """
        super().__init__(images, medium_megapix, low_megapix, final_megapix)
        self._names = Images.resolve_wildcards(images)
        self._names_set = True
        if len(self.names) < 2:
            raise StitchingError("2 or more Images needed")
        self._sizes = []

    def subset(self, indices):
        """
        获取子集
        :param indices: 索引列表
        """
        super().subset(indices)

    def __iter__(self):
        """
        迭代图像
        :return: 图像生成器
        """
        for idx, name in enumerate(self.names):
            img = Images.read_image(name)
            size = Images.get_image_size(img)

            # ------
            # Attention for side effects!
            # the scalers are set on the first run
            self._set_scales(size)

            # the original image sizes are set on the first run
            if not self._sizes_set:
                self._sizes.append(size)
                if idx + 1 == len(self.names):
                    self._sizes_set = True
            # ------

            yield img
