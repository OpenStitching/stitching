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
        assert self._sizes_set
        return self._sizes

    @property
    def names(self):
        assert self._names_set
        return self._names

    @abstractmethod
    def subset(self, indices):
        self._sizes = [self._sizes[i] for i in indices]
        self._names = [self._names[i] for i in indices]

    def resize(self, resolution, imgs=None):
        img_iterable = self.__iter__() if imgs is None else imgs
        for idx, img in enumerate(img_iterable):
            yield Images.resize_img_by_scaler(
                self._get_scaler(resolution), self._sizes[idx], img
            )

    @abstractmethod
    def __iter__(self):
        pass

    def _set_scales(self, size):
        if not self._scales_set:
            for scaler in self._scalers.values():
                scaler.set_scale_by_img_size(size)
            self._scales_set = True

    def _get_scaler(self, resolution):
        Images.check_resolution(resolution)
        return self._scalers[resolution.name]

    def get_ratio(self, from_resolution, to_resolution):
        assert self._scales_set
        Images.check_resolution(from_resolution)
        Images.check_resolution(to_resolution)
        return (
            self._get_scaler(to_resolution).scale
            / self._get_scaler(from_resolution).scale  # noqa: W503
        )

    def get_scaled_img_sizes(self, resolution):
        assert self._scales_set and self._sizes_set
        Images.check_resolution(resolution)
        return [
            self._get_scaler(resolution).get_scaled_img_size(sz) for sz in self._sizes
        ]

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
            img_names = [i for i in glob(img_names[0]) if not os.path.isdir(i)]
        return img_names

    @staticmethod
    def check_list_element_types(list_, type_):
        return all([isinstance(element, type_) for element in list_])

    @staticmethod
    def to_binary(img):
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(img, 0.5, 255.0, cv.THRESH_BINARY)
        return binary.astype(np.uint8)


class _NumpyImages(Images):
    def __init__(self, images, medium_megapix, low_megapix, final_megapix):
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
        super().subset(indices)
        self._images = [self._images[i] for i in indices]

    def __iter__(self):
        for img in self._images:
            yield img


class _FilenameImages(Images):
    def __init__(self, images, medium_megapix, low_megapix, final_megapix):
        super().__init__(images, medium_megapix, low_megapix, final_megapix)
        self._names = Images.resolve_wildcards(images)
        self._names_set = True
        if len(self.names) < 2:
            raise StitchingError("2 or more Images needed")
        self._sizes = []

    def subset(self, indices):
        super().subset(indices)

    def __iter__(self):
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
