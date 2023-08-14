import glob
from abc import ABC, abstractmethod
from enum import Enum

import cv2 as cv

from .megapix_scaler import MegapixDownscaler
from .stitching_error import StitchingError


class Images(ABC):
    class Resolution(Enum):
        MEDIUM = 0.6
        LOW = 0.1
        FINAL = -1

    @abstractmethod
    def __init__(self, images, medium_megapix, low_megapix, final_megapix):
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

    @property
    def sizes(self):
        return self._sizes

    @sizes.setter
    def food_eaten(self, sizes):
        self._sizes = sizes

    @property
    def names(self):
        return self._names

    @food_eaten.setter
    def food_eaten(self, names):
        self._names = names

    @abstractmethod
    def subset(self):
        pass

    def resize(self, resolution):
        for idx, img in enumerate(self.__iter__()):
            yield Images.resize_img_by_scaler(
                self._get_scaler(resolution), self._sizes[idx], img
            )

    def _set_scales(self, size):
        if not self._scales_set:
            for scaler in self._scalers.values():
                scaler.set_scale_by_img_size(size)
            self._scales_set = True

    def _get_scaler(self, resolution):
        Images.check_resolution(resolution)
        return self._scalers[resolution.name]

    def get_ratio(self, from_resolution, to_resolution):
        Images.check_resolution(from_resolution)
        Images.check_resolution(to_resolution)
        return (
            self._get_scaler(to_resolution).scale
            / self._get_scaler(from_resolution).scale  # noqa: W503
        )

    def get_scaled_img_sizes(self, resolution):
        Images.check_resolution(resolution)
        return [
            self._get_scaler(resolution).get_scaled_img_size(sz) for sz in self._sizes
        ]

    @abstractmethod
    def __iter__(self):
        pass

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

    @staticmethod
    def check_resolution(resolution):
        assert isinstance(resolution, Enum) and resolution in Images.Resolution

    @staticmethod
    def resolve_wildcards(img_names):
        if len(img_names) == 1:
            img_names = glob.glob(img_names[0])
        return img_names


class NumpyImages(Images):
    def __init__(
        self,
        images,
        medium_megapix=Images.Resolution.MEDIUM.value,
        low_megapix=Images.Resolution.LOW.value,
        final_megapix=Images.Resolution.FINAL.value,
    ):
        super().__init__(images, medium_megapix, low_megapix, final_megapix)
        if len(images) < 2:
            raise StitchingError("2 or more Images needed")
        self.images = images
        self._sizes = [Images.get_image_size(img) for img in images]
        self._names = [str(i + 1) for i in range(len(images))]
        self._set_scales(self._sizes[0])

    def subset(self):
        pass

    def __iter__(self):
        for img in self.images:
            yield img


class NamedImages(Images):
    def __init__(
        self,
        images,
        medium_megapix=Images.Resolution.MEDIUM.value,
        low_megapix=Images.Resolution.LOW.value,
        final_megapix=Images.Resolution.FINAL.value,
    ):
        super().__init__(images, medium_megapix, low_megapix, final_megapix)
        self._names = Images.resolve_wildcards(images)
        if len(self.names) < 2:
            raise StitchingError("2 or more Images needed")
        self._sizes = []
        self.sizes_set = False

    def subset(self):
        pass

    def __iter__(self):
        for idx, name in enumerate(self.names):
            img = Images.read_image(name)
            size = Images.get_image_size(img)

            # ------
            # Attention for side effects!
            # the scalers are set on the first run
            self._set_scales(size)

            # the original image sizes are set on the first run
            if not self.sizes_set:
                self._sizes.append(size)
                if idx + 1 == len(self.names):
                    self.sizes_set = True
            # ------

            yield img
