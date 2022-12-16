import logging

import numpy as np

from helper.get_attributes import get_attributes
from helper.run_and_measure_time import run_and_measure_time

from core.image import Image


class Rotator:
    def __init__(self, name: str):
        self.logger = logging.getLogger(f"eremore.{__name__}")
        self.name = name

    def rotate_k(self, image: Image, k: int):
        attributes = get_attributes(self)
        arguments = {'image': image, 'k': k}
        self.logger.debug(f"Rotating with -> attributes: {attributes} | arguments: {arguments}")
        run_and_measure_time(self._rotate_k, arguments, logger=self.logger)

    def _rotate_k(self, image: Image, k):
        image.raw_image = np.rot90(image.raw_image, k)

    def rotate_right(self, image: Image):
        self.rotate_k(image, 3)

    def rotate_left(self, image: Image):
        self.rotate_k(image, 1)
