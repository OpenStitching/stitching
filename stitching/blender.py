import cv2 as cv
import numpy as np


class Blender:
    """https://docs.opencv.org/4.x/d6/d4a/classcv_1_1detail_1_1Blender.html"""

    BLENDER_CHOICES = (
        "multiband",
        "feather",
        "no",
    )
    DEFAULT_BLENDER = "multiband"
    DEFAULT_BLEND_STRENGTH = 5

    def __init__(
        self, blender_type=DEFAULT_BLENDER, blend_strength=DEFAULT_BLEND_STRENGTH
    ):
        """
        初始化Blender类
        :param blender_type: 混合器类型
        :param blend_strength: 混合强度
        """
        self.blender_type = blender_type
        self.blend_strength = blend_strength
        self.blender = None

    def prepare(self, corners, sizes):
        """
        准备混合器
        :param corners: 图像的角点
        :param sizes: 图像的尺寸
        """
        dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
        blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * self.blend_strength / 100

        if self.blender_type == "no" or blend_width < 1:
            self.blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)

        elif self.blender_type == "multiband":
            self.blender = cv.detail_MultiBandBlender()
            self.blender.setNumBands(int((np.log(blend_width) / np.log(2.0) - 1.0)))

        elif self.blender_type == "feather":
            self.blender = cv.detail_FeatherBlender()
            self.blender.setSharpness(1.0 / blend_width)

        self.blender.prepare(dst_sz)

    def feed(self, img, mask, corner):
        """
        向混合器中添加图像
        :param img: 图像
        :param mask: 掩码
        :param corner: 角点
        """
        self.blender.feed(cv.UMat(img.astype(np.int16)), mask, corner)

    def blend(self):
        """
        混合图像
        :return: 混合后的图像和掩码
        """
        result = None
        result_mask = None
        result, result_mask = self.blender.blend(result, result_mask)
        result = cv.convertScaleAbs(result)
        return result, result_mask

    @classmethod
    def create_panorama(cls, imgs, masks, corners, sizes):
        """
        创建全景图
        :param imgs: 图像列表
        :param masks: 掩码列表
        :param corners: 角点列表
        :param sizes: 尺寸列表
        :return: 全景图和掩码
        """
        blender = cls("no")
        blender.prepare(corners, sizes)
        for img, mask, corner in zip(imgs, masks, corners):
            blender.feed(img, mask, corner)
        return blender.blend()
