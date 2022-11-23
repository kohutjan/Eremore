from abc import ABC, abstractmethod

import logging

import numpy as np
import numpy.typing as npt

from helper.run_and_measure_time import run_and_measure_time


class Rotator(ABC):
    def __init__(self):
        self.logger = logging.getLogger(f"eremore.{__name__}")

    def rotate_k(self, image, k) -> npt.NDArray:
        arguments = {'k': k}
        self.logger.debug(f"Rotating with: -> attributes: {vars(self)} | arguments: {arguments}")
        image, elapsed_time = run_and_measure_time(self._rotate_k,
                                                   {'image': image, 'k': k},
                                                   logger=self.logger)
        return image

    def rotate_right(self, image) -> npt.NDArray:
        return self.rotate_k(image, 3)

    def rotate_left(self, image) -> npt.NDArray:
        return self.rotate_k(image, 1)

    @abstractmethod
    def _rotate_k(self, image, k) -> npt.NDArray:
        pass


class RotatorBasic(Rotator):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(f"eremore.{__name__}.basic")
        self.name = "RotatorBasic"

    def _rotate_k(self, image, k):
        return np.rot90(image, k)
