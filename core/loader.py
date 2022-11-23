from abc import ABC, abstractmethod

import logging

import rawpy
import numpy as np
import numpy.typing as npt

from helper.run_and_measure_time import run_and_measure_time


class Loader(ABC):
    def __init__(self):
        self.logger = logging.getLogger(f"eremore.{__name__}")

    def load(self, path_to_raw_image: str) -> npt.NDArray[np.uint16]:
        arguments = {'path_to_raw_image': path_to_raw_image}
        self.logger.debug(f"Loading with -> attributes: {vars(self)} | arguments: {arguments}")
        image, elapsed_time = run_and_measure_time(self._load,
                                                   {'path_to_raw_image': path_to_raw_image},
                                                   logger=self.logger)
        return image

    @abstractmethod
    def _load(self, path_to_raw_image: str) -> npt.NDArray[np.uint16]:
        pass


class LoaderSimplePyRaw(Loader):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(f"eremore.{__name__}.simple_pyraw")
        self.name = "SimpleRawPyLoader"

    def _load(self, path_to_raw_image):
        with rawpy.imread(path_to_raw_image) as rawpy_loader:
            raw_image = rawpy_loader.raw_image.copy()
        return raw_image
