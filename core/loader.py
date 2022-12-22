import sys
from abc import ABC, abstractmethod

import logging

import rawpy
import numpy as np

from collections import OrderedDict

from helper.get_attributes import get_attributes
from helper.run_and_measure_time import run_and_measure_time

from core.image import Image


class Loader:
    def __init__(self, name: str = 'exporter', engine: str = 'open_cv'):
        self.logger = logging.getLogger(f"eremore.{__name__}")
        self.name = name
        self.engine = engine
        self.engines = OrderedDict()
        self.engines['raw_py'] = LoaderRawPy()

    def process(self, image: Image = None):
        if self.engine not in self.engines.keys():
            self.logger.error(f"Exporter engine {self.engine} does not exists.")
            raise ValueError

        return self.engines[self.engine].load()


class LoaderBase(ABC):
    def __init__(self):
        self.logger = logging.getLogger(f"eremore.{__name__}")
        self.name = None
        self.path_to_raw_image = None

    def load(self) -> Image:
        if self.path_to_raw_image is None:
            self.logger.warning(f"path_to_raw_image not set, returning placeholder.")
            return Image(raw_image=np.zeros((64, 64)))
        attributes = get_attributes(self)
        self.logger.debug(f"Loading with -> attributes: {attributes}")
        image, elapsed_time = run_and_measure_time(self._load, {}, logger=self.logger)
        return image

    @abstractmethod
    def _load(self) -> Image:
        pass

    def set(self, name=None, path_to_raw_image=None):
        self._set(name, path_to_raw_image)

    def _set(self, name=None, path_to_raw_image=None):
        if name is not None:
            self.name = name
            self.logger = logging.getLogger(f"eremore.{__name__}.{name}")
        if path_to_raw_image is not None:
            self.path_to_raw_image = path_to_raw_image


class LoaderRawPy(LoaderBase):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(f"eremore.{__name__}.rawpy")
        self.name = "RawPyLoader"

    def _load(self):
        with rawpy.imread(self.path_to_raw_image) as rawpy_loader:
            raw_image = rawpy_loader.raw_image.copy()
            image = Image(raw_image.astype(np.uint16),
                          camera_white_balance=np.asarray(rawpy_loader.camera_whitebalance[:3], dtype=np.float32))
        return image
