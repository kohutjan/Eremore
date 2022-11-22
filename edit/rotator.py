from abc import ABC, abstractmethod

import logging

import numpy as np
import numpy.typing as npt

from timeit import default_timer as timer

logger = logging.getLogger(f"eremore.{__name__}")


class Rotator(ABC):

    def rotate_k(self, image, k) -> npt.NDArray:
        arguments = {'k': k}
        logger.debug(f"Rotating with: -> attributes: {vars(self)} | arguments: {arguments}")
        start = timer()
        image = self._rotate_k(image, k=k)
        end = timer()
        elapsed_time = end - start
        logger.debug(f"Elapsed time: {elapsed_time}")
        return image

    def rotate_right(self, image) -> npt.NDArray:
        return self.rotate_k(image, 3)

    def rotate_left(self, image) -> npt.NDArray:
        return self.rotate_k(image, 1)

    @abstractmethod
    def _rotate_k(self, image, k) -> npt.NDArray:
        pass


class BasicRotator(Rotator):
    def __init__(self):
        self.name = "BasicRotator"

    def _rotate_k(self, image, k):
        return np.rot90(image, k)
