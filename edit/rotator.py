from abc import ABC, abstractmethod

import logging

import numpy as np

from helper.get_attributes import get_attributes
from helper.run_and_measure_time import run_and_measure_time

from core.image import Image


class Rotator(ABC):
    def __init__(self):
        self.logger = logging.getLogger(f"eremore.{__name__}")

    def rotate_k(self, image: Image, k: int):
        attributes = get_attributes(self)
        arguments = {'image': image, 'k': k}
        self.logger.debug(f"Rotating with: -> attributes: {attributes} | arguments: {arguments}")
        run_and_measure_time(self._rotate_k, arguments, logger=self.logger)

    @abstractmethod
    def _rotate_k(self, image, k):
        pass

    def rotate_right(self, image):
        self.rotate_k(image.raw_image, 3)

    def rotate_left(self, image):
        self.rotate_k(image.raw_image, 1)


class RotatorNumPy90(Rotator):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(f"eremore.{__name__}.basic")
        self.name = "RotatorBasic"

    def _rotate_k(self, image, k):
        image.raw_image = np.rot90(image.raw_image, k)
