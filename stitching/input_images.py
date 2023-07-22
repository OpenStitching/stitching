import enum
import glob

import cv2 as cv

from .megapix_scaler import MegapixDownscaler
from .stitching_error import StitchingError


class InputImages:
    DEFAULT_MEDIUM_MEGAPIX = 0.6
    DEFAULT_LOW_MEGAPIX = 0.1
    DEFAULT_FINAL_MEGAPIX = -1

    def __init__(self, img_names):
        if len(img_names) == 1:
            img_names = glob.glob(img_names[0])
        if len(img_names) < 2:
            raise StitchingError("2 or more Images needed")
        self.img_names = img_names
        self.resolutions_set = False
        self.img_sizes = []
        self.img_sizes_set = False
        self.scalers = {}
        self.scales_set = False

    def set_resolutions(
        self,
        medium_megapix=DEFAULT_MEDIUM_MEGAPIX,
        low_megapix=DEFAULT_LOW_MEGAPIX,
        final_megapix=DEFAULT_FINAL_MEGAPIX,
    ):
        assert not self.resolutions_set
        if medium_megapix < low_megapix:
            raise StitchingError(
                "Medium resolution megapix need to be "
                "greater or equal than low resolution "
                "megapix"
            )
        self.resolutions = enum.Enum(
            "Resolutions",
            {"MEDIUM": medium_megapix, "LOW": low_megapix, "FINAL": final_megapix},
        )
        self.resolutions_set = True

    def read_and_resize(self, resolution):
        assert self.resolutions_set
        assert isinstance(resolution, enum.Enum)
        img_sizes = []
        for idx, name in enumerate(self.img_names):
            img = InputImages.read_image(name)
            size = InputImages.get_image_size(img)

            # ------
            # Attention for side effects!
            # the scalers are set on the first run
            if not self.scales_set:
                self.scalers["MEDIUM"] = MegapixDownscaler(
                    self.resolutions.MEDIUM.value
                )
                self.scalers["LOW"] = MegapixDownscaler(self.resolutions.LOW.value)
                self.scalers["FINAL"] = MegapixDownscaler(self.resolutions.FINAL.value)
                for scaler in self.scalers.values():
                    scaler.set_scale_by_img_size(size)
                self.scales_set = True

            # and the original img_sizes are set on the first run
            img_sizes.append(size)
            if not self.img_sizes_set and idx + 1 == len(self.img_names):
                self.img_sizes = img_sizes
                self.img_sizes_set = True
            # ------

            yield InputImages.resize_img_by_scaler(
                self.get_scaler(resolution), size, img
            )

    def resize(self, imgs, resolution):
        assert self.scales_set
        assert isinstance(resolution, enum.Enum)
        for img, size in zip(imgs, self.img_sizes):
            yield InputImages.resize_img_by_scaler(
                self.get_scaler(resolution), size, img
            )

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
        assert isinstance(resolution, enum.Enum)
        return self.scalers[resolution.name]

    def get_ratio(self, from_resolution, to_resolution):
        assert isinstance(from_resolution, enum.Enum)
        assert isinstance(to_resolution, enum.Enum)
        return (
            self.get_scaler(to_resolution).scale
            / self.get_scaler(from_resolution).scale  # noqa: W503
        )

    def get_scaled_img_sizes(self, resolution):
        assert isinstance(resolution, enum.Enum)
        return [
            self.get_scaler(resolution).get_scaled_img_size(sz) for sz in self.img_sizes
        ]
