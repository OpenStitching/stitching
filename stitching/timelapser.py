import os

import cv2 as cv
import numpy as np


class Timelapser:
    """https://docs.opencv.org/4.x/dd/dac/classcv_1_1detail_1_1Timelapser.html"""

    TIMELAPSE_CHOICES = (
        "no",
        "as_is",
        "crop",
    )
    DEFAULT_TIMELAPSE = "no"
    DEFAULT_TIMELAPSE_PREFIX = "fixed_"

    def __init__(
        self, timelapse=DEFAULT_TIMELAPSE, timelapse_prefix=DEFAULT_TIMELAPSE_PREFIX
    ):
        self.do_timelapse = True
        self.timelapse_type = None
        self.timelapser = None
        self.timelapse_prefix = timelapse_prefix

        if timelapse == "as_is":
            self.timelapse_type = cv.detail.Timelapser_AS_IS
        elif timelapse == "crop":
            self.timelapse_type = cv.detail.Timelapser_CROP
        else:
            self.do_timelapse = False

        if self.do_timelapse:
            self.timelapser = cv.detail.Timelapser_createDefault(self.timelapse_type)

    def initialize(self, *args):
        self.timelapser.initialize(*args)

    def process_and_save_frame(self, img_name, img, corner):
        self.process_frame(img, corner)
        cv.imwrite(self.get_fixed_filename(img_name), self.get_frame())

    def process_frame(self, img, corner):
        mask = np.ones((img.shape[0], img.shape[1]), np.uint8)
        img = img.astype(np.int16)
        self.timelapser.process(img, mask, corner)

    def get_frame(self):
        frame = self.timelapser.getDst()
        frame = np.float32(cv.UMat.get(frame))
        frame = cv.convertScaleAbs(frame)
        return frame

    def get_fixed_filename(self, img_name):
        dirname, filename = os.path.split(img_name)
        return os.path.join(dirname, self.timelapse_prefix + filename)
