from collections import namedtuple

import cv2 as cv
import numpy as np

from .blender import Blender
from .stitching_error import StitchingError


class Rectangle(namedtuple("Rectangle", "x y width height")):
    __slots__ = ()

    @property
    def area(self):
        return self.width * self.height

    @property
    def corner(self):
        return (self.x, self.y)

    @property
    def size(self):
        return (self.width, self.height)

    @property
    def x2(self):
        return self.x + self.width

    @property
    def y2(self):
        return self.y + self.height

    def times(self, x):
        return Rectangle(*(int(round(i * x)) for i in self))

    def draw_on(self, img, color=(0, 0, 255), size=1):
        if len(img.shape) == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        start_point = (self.x, self.y)
        end_point = (self.x2 - 1, self.y2 - 1)
        cv.rectangle(img, start_point, end_point, color, size)
        return img


class Cropper:
    DEFAULT_CROP = True

    def __init__(self, crop=DEFAULT_CROP):
        """
        初始化Cropper类
        :param crop: 是否进行裁剪
        """
        self.do_crop = crop
        self.overlapping_rectangles = []
        self.cropping_rectangles = []

    def prepare(self, imgs, masks, corners, sizes):
        """
        准备裁剪器
        :param imgs: 图像列表
        :param masks: 掩码列表
        :param corners: 角点列表
        :param sizes: 尺寸列表
        """
        if self.do_crop:
            mask = self.estimate_panorama_mask(imgs, masks, corners, sizes)
            lir = self.estimate_largest_interior_rectangle(mask)
            corners = self.get_zero_center_corners(corners)
            rectangles = self.get_rectangles(corners, sizes)
            self.overlapping_rectangles = self.get_overlaps(rectangles, lir)
            self.intersection_rectangles = self.get_intersections(
                rectangles, self.overlapping_rectangles
            )

    def crop_images(self, imgs, aspect=1):
        """
        裁剪图像
        :param imgs: 图像列表
        :param aspect: 缩放比例
        :return: 裁剪后的图像生成器
        """
        for idx, img in enumerate(imgs):
            yield self.crop_img(img, idx, aspect)

    def crop_img(self, img, idx, aspect=1):
        """
        裁剪单张图像
        :param img: 图像
        :param idx: 图像索引
        :param aspect: 缩放比例
        :return: 裁剪后的图像
        """
        if self.do_crop:
            intersection_rect = self.intersection_rectangles[idx]
            scaled_intersection_rect = intersection_rect.times(aspect)
            cropped_img = self.crop_rectangle(img, scaled_intersection_rect)
            return cropped_img
        return img

    def crop_rois(self, corners, sizes, aspect=1):
        """
        裁剪感兴趣区域（ROI）
        :param corners: 角点列表
        :param sizes: 尺寸列表
        :param aspect: 缩放比例
        :return: 裁剪后的角点和尺寸
        """
        if self.do_crop:
            scaled_overlaps = [r.times(aspect) for r in self.overlapping_rectangles]
            cropped_corners = [r.corner for r in scaled_overlaps]
            cropped_corners = self.get_zero_center_corners(cropped_corners)
            cropped_sizes = [r.size for r in scaled_overlaps]
            return cropped_corners, cropped_sizes
        return corners, sizes

    @staticmethod
    def estimate_panorama_mask(imgs, masks, corners, sizes):
        """
        估计全景图掩码
        :param imgs: 图像列表
        :param masks: 掩码列表
        :param corners: 角点列表
        :param sizes: 尺寸列表
        :return: 全景图掩码
        """
        _, mask = Blender.create_panorama(imgs, masks, corners, sizes)
        return mask

    def estimate_largest_interior_rectangle(self, mask):
        """
        估计最大内部矩形
        :param mask: 掩码
        :return: 最大内部矩形
        """
        # largestinteriorrectangle is only imported if cropping
        # is explicitly desired (needs some time to compile at the first run!)
        import largestinteriorrectangle

        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        if not hierarchy.shape == (1, 1, 4) or not np.all(hierarchy == -1):
            raise StitchingError(
                "Invalid Contour. Run with --no-crop (using the stitch interface), crop=false (using the stitcher class) or Cropper(False) (using the cropper class)"  # noqa: E501
            )
        contour = contours[0][:, 0, :]

        lir = largestinteriorrectangle.lir(mask > 0, contour)
        lir = Rectangle(*lir)
        return lir

    @staticmethod
    def get_zero_center_corners(corners):
        """
        获取以零为中心的角点
        :param corners: 角点列表
        :return: 以零为中心的角点列表
        """
        min_corner_x = min([corner[0] for corner in corners])
        min_corner_y = min([corner[1] for corner in corners])
        return [(x - min_corner_x, y - min_corner_y) for x, y in corners]

    @staticmethod
    def get_rectangles(corners, sizes):
        """
        获取矩形列表
        :param corners: 角点列表
        :param sizes: 尺寸列表
        :return: 矩形列表
        """
        rectangles = []
        for corner, size in zip(corners, sizes):
            rectangle = Rectangle(*corner, *size)
            rectangles.append(rectangle)
        return rectangles

    @staticmethod
    def get_overlaps(rectangles, lir):
        """
        获取重叠矩形列表
        :param rectangles: 矩形列表
        :param lir: 最大内部矩形
        :return: 重叠矩形列表
        """
        return [Cropper.get_overlap(r, lir) for r in rectangles]

    @staticmethod
    def get_overlap(rectangle1, rectangle2):
        """
        获取两个矩形的重叠部分
        :param rectangle1: 矩形1
        :param rectangle2: 矩形2
        :return: 重叠部分矩形
        """
        x1 = max(rectangle1.x, rectangle2.x)
        y1 = max(rectangle1.y, rectangle2.y)
        x2 = min(rectangle1.x2, rectangle2.x2)
        y2 = min(rectangle1.y2, rectangle2.y2)
        if x2 < x1 or y2 < y1:
            raise StitchingError("Rectangles do not overlap!")
        return Rectangle(x1, y1, x2 - x1, y2 - y1)

    @staticmethod
    def get_intersections(rectangles, overlapping_rectangles):
        """
        获取矩形与重叠矩形的交集
        :param rectangles: 矩形列表
        :param overlapping_rectangles: 重叠矩形列表
        :return: 交集矩形列表
        """
        return [
            Cropper.get_intersection(r, overlap_r)
            for r, overlap_r in zip(rectangles, overlapping_rectangles)
        ]

    @staticmethod
    def get_intersection(rectangle, overlapping_rectangle):
        """
        获取矩形与重叠矩形的交集
        :param rectangle: 矩形
        :param overlapping_rectangle: 重叠矩形
        :return: 交集矩形
        """
        x = abs(overlapping_rectangle.x - rectangle.x)
        y = abs(overlapping_rectangle.y - rectangle.y)
        width = overlapping_rectangle.width
        height = overlapping_rectangle.height
        return Rectangle(x, y, width, height)

    @staticmethod
    def crop_rectangle(img, rectangle):
        """
        裁剪矩形区域
        :param img: 图像
        :param rectangle: 矩形区域
        :return: 裁剪后的图像
        """
        return img[rectangle.y : rectangle.y2, rectangle.x : rectangle.x2]
