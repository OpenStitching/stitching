import warnings
from collections import OrderedDict

import cv2 as cv
import numpy as np

from .blender import Blender
from .stitching_error import StitchingWarning


class SeamFinder:
    """https://docs.opencv.org/4.x/d7/d09/classcv_1_1detail_1_1SeamFinder.html"""

    SEAM_FINDER_CHOICES = OrderedDict()
    SEAM_FINDER_CHOICES["dp_color"] = cv.detail_DpSeamFinder("COLOR")
    SEAM_FINDER_CHOICES["dp_colorgrad"] = cv.detail_DpSeamFinder("COLOR_GRAD")
    SEAM_FINDER_CHOICES["gc_color"] = cv.detail_GraphCutSeamFinder("COST_COLOR")
    SEAM_FINDER_CHOICES["gc_colorgrad"] = cv.detail_GraphCutSeamFinder(
        "COST_COLOR_GRAD"
    )
    SEAM_FINDER_CHOICES["voronoi"] = cv.detail.SeamFinder_createDefault(
        cv.detail.SeamFinder_VORONOI_SEAM
    )
    SEAM_FINDER_CHOICES["no"] = cv.detail.SeamFinder_createDefault(
        cv.detail.SeamFinder_NO
    )

    DEFAULT_SEAM_FINDER = list(SEAM_FINDER_CHOICES.keys())[0]

    def __init__(self, finder=DEFAULT_SEAM_FINDER):
        """
        初始化SeamFinder类
        :param finder: 接缝查找器类型
        """
        self.finder = SeamFinder.SEAM_FINDER_CHOICES[finder]

    def find(self, imgs, corners, masks):
        """
        查找接缝
        :param imgs: 图像列表
        :param corners: 角点列表
        :param masks: 掩码列表
        :return: 接缝掩码列表
        """
        imgs_float = [img.astype(np.float32) for img in imgs]
        return self.finder.find(imgs_float, corners, masks)

    @staticmethod
    def resize(seam_mask, mask):
        """
        调整接缝掩码大小
        :param seam_mask: 接缝掩码
        :param mask: 掩码
        :return: 调整大小后的接缝掩码
        """
        dilated_mask = cv.dilate(seam_mask, None)
        resized_seam_mask = cv.resize(
            dilated_mask, (mask.shape[1], mask.shape[0]), 0, 0, cv.INTER_LINEAR_EXACT
        )
        return cv.bitwise_and(resized_seam_mask, mask)

    @staticmethod
    def draw_seam_mask(img, seam_mask, color=(0, 0, 0)):
        """
        在图像上绘制接缝掩码
        :param img: 图像
        :param seam_mask: 接缝掩码
        :param color: 颜色
        :return: 绘制了接缝掩码的图像
        """
        seam_mask = cv.UMat.get(seam_mask)
        overlaid_img = np.copy(img)
        overlaid_img[seam_mask == 0] = color
        return overlaid_img

    @staticmethod
    def draw_seam_polygons(panorama, blended_seam_masks, alpha=0.5):
        """
        在全景图上绘制接缝多边形
        :param panorama: 全景图
        :param blended_seam_masks: 混合接缝掩码
        :param alpha: 透明度
        :return: 绘制了接缝多边形的全景图
        """
        return add_weighted_image(panorama, blended_seam_masks, alpha)

    @staticmethod
    def draw_seam_lines(panorama, blended_seam_masks, linesize=1, color=(0, 0, 255)):
        """
        在全景图上绘制接缝线
        :param panorama: 全景图
        :param blended_seam_masks: 混合接缝掩码
        :param linesize: 线条宽度
        :param color: 颜色
        :return: 绘制了接缝线的全景图
        """
        seam_lines = SeamFinder.extract_seam_lines(blended_seam_masks, linesize)
        panorama_with_seam_lines = panorama.copy()
        panorama_with_seam_lines[seam_lines == 255] = color
        return panorama_with_seam_lines

    @staticmethod
    def extract_seam_lines(blended_seam_masks, linesize=1):
        """
        提取接缝线
        :param blended_seam_masks: 混合接缝掩码
        :param linesize: 线条宽度
        :return: 接缝线
        """
        seam_lines = cv.Canny(np.uint8(blended_seam_masks), 100, 200)
        seam_indices = (seam_lines == 255).nonzero()
        seam_lines = remove_invalid_line_pixels(
            seam_indices, seam_lines, blended_seam_masks
        )
        kernelsize = linesize + linesize - 1
        kernel = np.ones((kernelsize, kernelsize), np.uint8)
        return cv.dilate(seam_lines, kernel)

    @staticmethod
    def blend_seam_masks(
        seam_masks,
        corners,
        sizes,
        colors=(
            (255, 000, 000),  # 红色
            (000, 000, 255),  # 蓝色
            (000, 255, 000),  # 绿色
            (000, 255, 255),  # 黄色
            (255, 000, 255),  # 紫色
            (128, 128, 255),  # 粉色
            (128, 128, 128),  # 灰色
            (000, 000, 128),  # 深蓝色
            (000, 128, 255),  # 浅蓝色
        ),
    ):
        """
        混合接缝掩码
        :param seam_masks: 接缝掩码列表
        :param corners: 角点列表
        :param sizes: 尺寸列表
        :param colors: 颜色列表
        :return: 混合后的接缝掩码
        """
        imgs = colored_img_generator(sizes, colors)
        blended_seam_masks, _ = Blender.create_panorama(
            imgs, seam_masks, corners, sizes
        )
        return blended_seam_masks


def colored_img_generator(sizes, colors):
    """
    生成彩色图像
    :param sizes: 尺寸列表
    :param colors: 颜色列表
    :return: 彩色图像生成器
    """
    if len(sizes) + 1 > len(colors):
        warnings.warn(
            "没有额外的颜色，将有接缝掩码具有相同的颜色",  # noqa: E501
            StitchingWarning,
        )

    for idx, size in enumerate(sizes):
        yield create_img_by_size(size, colors[idx % len(colors)])


def create_img_by_size(size, color=(0, 0, 0)):
    """
    根据尺寸创建图像
    :param size: 尺寸
    :param color: 颜色
    :return: 创建的图像
    """
    width, height = size
    img = np.zeros((height, width, 3), np.uint8)
    img[:] = color
    return img


def add_weighted_image(img1, img2, alpha):
    """
    添加加权图像
    :param img1: 图像1
    :param img2: 图像2
    :param alpha: 透明度
    :return: 加权后的图像
    """
    return cv.addWeighted(img1, alpha, img2, (1.0 - alpha), 0.0)


def remove_invalid_line_pixels(indices, lines, mask):
    """
    移除无效的线条像素
    :param indices: 索引
    :param lines: 线条
    :param mask: 掩码
    :return: 移除无效线条像素后的线条
    """
    for x, y in zip(*indices):
        if check_if_pixel_or_neighbor_is_black(mask, x, y):
            lines[x, y] = 0
    return lines


def check_if_pixel_or_neighbor_is_black(img, x, y):
    """
    检查像素或邻居是否为黑色
    :param img: 图像
    :param x: x坐标
    :param y: y坐标
    :return: 是否为黑色
    """
    check = [
        is_pixel_black(img, x, y),
        is_pixel_black(img, x + 1, y),
        is_pixel_black(img, x - 1, y),
        is_pixel_black(img, x, y + 1),
        is_pixel_black(img, x, y - 1),
    ]
    return any(check)


def is_pixel_black(img, x, y):
    """
    检查像素是否为黑色
    :param img: 图像
    :param x: x坐标
    :param y: y坐标
    :return: 是否为黑色
    """
    return np.all(get_pixel_value(img, x, y) == 0)


def get_pixel_value(img, x, y):
    """
    获取像素值
    :param img: 图像
    :param x: x坐标
    :param y: y坐标
    :return: 像素值
    """
    try:
        return img[x, y]
    except IndexError:
        pass
