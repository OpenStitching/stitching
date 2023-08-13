import glob
from enum import Enum

import cv2 as cv

from .megapix_scaler import MegapixDownscaler
from .stitching_error import StitchingError


class Images:
    class Resolution(Enum):
        MEDIUM = 0.6
        LOW = 0.1
        FINAL = -1

    def __init__(
        self,
        img_names,
        medium_megapix=Resolution.MEDIUM.value,
        low_megapix=Resolution.LOW.value,
        final_megapix=Resolution.FINAL.value,
    ):
        self.img_names = Images.resolve_wildcards(img_names)
        if len(self.img_names) < 2:
            raise StitchingError("2 or more Images needed")

        if medium_megapix < low_megapix:
            raise StitchingError(
                "Medium resolution megapix need to be "
                "greater or equal than low resolution "
                "megapix"
            )

        self.scalers = {}
        self.scalers["MEDIUM"] = MegapixDownscaler(medium_megapix)
        self.scalers["LOW"] = MegapixDownscaler(low_megapix)
        self.scalers["FINAL"] = MegapixDownscaler(final_megapix)

        self.img_sizes = []
        self.img_sizes_set = False
        self.scales_set = False

    def read(self):
        for idx, name in enumerate(self.img_names):
            img = Images.read_image(name)
            size = Images.get_image_size(img)

            # ------
            # Attention for side effects!
            # the scalers are set on the first run
            if not self.scales_set:
                for scaler in self.scalers.values():
                    scaler.set_scale_by_img_size(size)
                self.scales_set = True

            # and the original img_sizes are set on the first run
            if not self.img_sizes_set:
                self.img_sizes.append(size)
                if idx + 1 == len(self.img_names):
                    self.img_sizes_set = True
            # ------

            yield img

    def resize(self, imgs, resolution):
        Images.check_resolution(resolution)
        for idx, img in enumerate(imgs):
            yield Images.resize_img_by_scaler(
                self.get_scaler(resolution), self.img_sizes[idx], img
            )

    @staticmethod
    def resolve_wildcards(img_names):
        if len(img_names) == 1:
            img_names = glob.glob(img_names[0])
        return img_names

    @staticmethod
    def read_image(img_name):
        img = cv.imread(img_name)
        if img is None:
            raise StitchingError("Cannot read image " + img_name)
        return img

    @staticmethod
    def get_image_size(img):
        """(width, height)"""
        return (img.shape[1], img.shape[0])

    @staticmethod
    def resize_img_by_scaler(scaler, size, img):
        desired_size = scaler.get_scaled_img_size(size)
        return cv.resize(img, desired_size, interpolation=cv.INTER_LINEAR_EXACT)

    def get_scaler(self, resolution):
        Images.check_resolution(resolution)
        return self.scalers[resolution.name]

    def get_ratio(self, from_resolution, to_resolution):
        Images.check_resolution(from_resolution)
        Images.check_resolution(to_resolution)
        return (
            self.get_scaler(to_resolution).scale
            / self.get_scaler(from_resolution).scale  # noqa: W503
        )

    def get_scaled_img_sizes(self, resolution):
        Images.check_resolution(resolution)
        return [
            self.get_scaler(resolution).get_scaled_img_size(sz) for sz in self.img_sizes
        ]

    def check_resolution(resolution):
        assert isinstance(resolution, Enum) and resolution in Images.Resolution