from abc import ABC, abstractmethod

import logging

import rawpy
import numpy as np
import numpy.typing as npt

from timeit import default_timer as timer

logger = logging.getLogger(f"eremore.{__name__}")


class Loader(ABC):
    def load(self, path_to_raw_image: str) -> npt.NDArray[np.uint16]:
        arguments = {'path_to_raw_image': path_to_raw_image}
        logger.debug(f"Loading with -> attributes: {vars(self)} | arguments: {arguments}")
        start = timer()
        image = self._load(path_to_raw_image)
        end = timer()
        elapsed_time = end - start
        logger.debug(f"Elapsed time: {elapsed_time}")
        return image
        pass

    @abstractmethod
    def _load(self, path_to_raw_image: str) -> npt.NDArray[np.uint16]:
        pass


class SimpleRawPyLoader(Loader):
    def __init__(self):
        self.name = "SimpleRawPyLoader"

    def _load(self, path_to_raw_image):
        with rawpy.imread(path_to_raw_image) as rawpy_loader:
            raw_image = rawpy_loader.raw_image.copy()
        return raw_image
