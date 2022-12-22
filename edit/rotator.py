from abc import ABC, abstractmethod

import logging

import numpy as np

from collections import OrderedDict

from helper.get_attributes import get_attributes
from helper.run_and_measure_time import run_and_measure_time

from core.image import Image


class Rotator:
    def __init__(self, name: str = 'rotator', engine: str = None):
        self.logger = logging.getLogger(f"eremore.{__name__}")
        self.name = name
        self.engine = engine
        self.engines = OrderedDict()
        self.engines['90'] = Rotator90()

    def process(self, image: Image):
        if self.engine is None:
            return
        if self.engine not in self.engines.keys():
            self.logger.error(f"Rotator engine {self.engine} does not exists.")
            raise ValueError

        self.engines[self.engine].rotate(image)


class RotatorBase(ABC):
    def __init__(self):
        self.logger = logging.getLogger(f"eremore.{__name__}")
        self.name = None

    def rotate(self, image: Image):
        attributes = get_attributes(self)
        arguments = {'image': image}
        self.logger.debug(f"Rotating with -> attributes: {attributes} | arguments: {arguments}")
        run_and_measure_time(self._rotate, arguments, logger=self.logger)

    @abstractmethod
    def _rotate(self, image: Image):
        pass

    def set(self, name=None):
        self._set(name)

    def _set(self, name=None):
        if name is not None:
            self.name = name
            self.logger = logging.getLogger(f"eremore.{__name__}.{name}")


class Rotator90(RotatorBase):
    def __init__(self, name='rotator_90', k: int = 1):
        super().__init__()
        self.logger = logging.getLogger(f"eremore.{__name__}.{name}")
        self.name = name
        self.k = k

    def _rotate(self, image: Image):
        image.raw_image = np.rot90(image.raw_image, self.k)

    def set(self, name=None, k=None):
        super()._set(name)
        if k is not None:
            self.k = k
