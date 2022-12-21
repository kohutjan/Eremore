from abc import ABC, abstractmethod

import logging

import rawpy
import numpy as np

from helper.get_attributes import get_attributes
from helper.run_and_measure_time import run_and_measure_time

from core.image import Image


class Loader(ABC):
    def __init__(self):
        self.logger = logging.getLogger(f"eremore.{__name__}")

    def load(self, path_to_raw_image: str) -> Image:
        attributes = get_attributes(self)
        arguments = {'path_to_raw_image': path_to_raw_image}
        self.logger.debug(f"Loading with -> attributes: {attributes} | arguments: {arguments}")
        image, elapsed_time = run_and_measure_time(self._load, arguments, logger=self.logger)
        return image

    @abstractmethod
    def _load(self, path_to_raw_image: str) -> Image:
        pass


class LoaderRawPy(Loader):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(f"eremore.{__name__}.rawpy")
        self.name = "RawPyLoader"

    def _load(self, path_to_raw_image):
        with rawpy.imread(path_to_raw_image) as rawpy_loader:
            raw_image = rawpy_loader.raw_image.copy()
            image = Image(raw_image.astype(np.uint16),
                          camera_white_balance=np.asarray(rawpy_loader.camera_whitebalance[:3], dtype=np.float32))
        return image
