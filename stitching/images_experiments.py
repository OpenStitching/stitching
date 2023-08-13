import glob
from abc import ABC, abstractmethod

import cv2 as cv

from .stitching_error import StitchingError


class Images(ABC):
    @abstractmethod
    def __init__(self, images):
        pass

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


class NumpyImages(Images):
    def __init__(self, images):
        if len(images) < 2:
            raise StitchingError("2 or more Images needed")
        self.images = images
        self._sizes = [Images.get_image_size(img) for img in images]
        self._names = [str(i + 1) for i in range(len(images))]

    def subset(self):
        pass

    def __iter__(self):
        for img in self.images:
            yield img


class NamedImages(Images):
    def __init__(self, images):
        self._names = self.resolve_wildcards(images)
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
            # the original image sizes are set on the first run
            if not self.sizes_set:
                self._sizes.append(size)
                if idx + 1 == len(self.names):
                    self.sizes_set = True
            # ------

            yield img

    @staticmethod
    def resolve_wildcards(img_names):
        if len(img_names) == 1:
            img_names = glob.glob(img_names[0])
        return img_names


Images.register(NumpyImages)
Images.register(NamedImages)
