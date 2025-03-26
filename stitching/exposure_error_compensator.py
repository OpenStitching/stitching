from collections import OrderedDict

import cv2 as cv


class ExposureErrorCompensator:
    """https://docs.opencv.org/4.x/d2/d37/classcv_1_1detail_1_1ExposureCompensator.html"""  # noqa: E501

    COMPENSATOR_CHOICES = OrderedDict()
    COMPENSATOR_CHOICES["gain_blocks"] = cv.detail.ExposureCompensator_GAIN_BLOCKS
    COMPENSATOR_CHOICES["gain"] = cv.detail.ExposureCompensator_GAIN
    COMPENSATOR_CHOICES["channel"] = cv.detail.ExposureCompensator_CHANNELS
    COMPENSATOR_CHOICES["channel_blocks"] = (
        cv.detail.ExposureCompensator_CHANNELS_BLOCKS
    )
    COMPENSATOR_CHOICES["no"] = cv.detail.ExposureCompensator_NO

    DEFAULT_COMPENSATOR = list(COMPENSATOR_CHOICES.keys())[0]
    DEFAULT_NR_FEEDS = 1
    DEFAULT_BLOCK_SIZE = 32

    def __init__(
        self,
        compensator=DEFAULT_COMPENSATOR,
        nr_feeds=DEFAULT_NR_FEEDS,
        block_size=DEFAULT_BLOCK_SIZE,
    ):
        """
        初始化ExposureErrorCompensator类
        :param compensator: 曝光补偿器类型
        :param nr_feeds: 曝光补偿器的数量
        :param block_size: 块大小
        """
        if compensator == "channel":
            self.compensator = cv.detail_ChannelsCompensator(nr_feeds)
        elif compensator == "channel_blocks":
            self.compensator = cv.detail_BlocksChannelsCompensator(
                block_size, block_size, nr_feeds
            )
        else:
            self.compensator = cv.detail.ExposureCompensator_createDefault(
                ExposureErrorCompensator.COMPENSATOR_CHOICES[compensator]
            )

    def feed(self, *args):
        """
        向曝光补偿器提供数据
        :param args: 数据参数
        """
        self.compensator.feed(*args)

    def apply(self, *args):
        """
        应用曝光补偿
        :param args: 数据参数
        :return: 曝光补偿后的结果
        """
        return self.compensator.apply(*args)
