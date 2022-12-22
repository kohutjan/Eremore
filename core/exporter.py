from abc import ABC, abstractmethod

import logging

import numpy as np
import cv2

from collections import OrderedDict

from helper.get_attributes import get_attributes
from helper.run_and_measure_time import run_and_measure_time

from core.image import Image


class Exporter:
    def __init__(self, name: str = 'exporter', engine: str = 'open_cv'):
        self.logger = logging.getLogger(f"eremore.{__name__}")
        self.name = name
        self.engine = engine
        self.engines = OrderedDict()
        self.engines['open_cv'] = ExporterOpenCV()

    def process(self, image: Image):
        if self.engine not in self.engines.keys():
            self.logger.error(f"Exporter engine {self.engine} does not exists.")
            raise ValueError

        self.engines[self.engine].export(image)


class ExporterBase(ABC):
    def __init__(self,
                 path_to_export_image: str = None):
        self.logger = logging.getLogger(f"eremore.{__name__}")
        self.name = None
        self.path_to_export_image = path_to_export_image

    def export(self, image: Image):
        if self.path_to_export_image is None:
            self.logger.warning(f"path_to_export_image not set, doing nothing.")
            return
        attributes = get_attributes(self)
        arguments = {'image': image}
        self.logger.debug(f"Exporting with -> attributes: {attributes} | arguments: {arguments}")
        run_and_measure_time(self._export, arguments, logger=self.logger)

    @abstractmethod
    def _export(self, image):
        pass

    def set(self, name=None, path_to_export_image=None):
        self._set(name, path_to_export_image)

    def _set(self, name=None, path_to_export_image=None):
        if name is not None:
            self.name = name
            self.logger = logging.getLogger(f"eremore.{__name__}.{name}")
        if path_to_export_image is not None:
            self.path_to_export_image = path_to_export_image


class ExporterOpenCV(ExporterBase):
    def __init__(self, name='open_cv'):
        super().__init__()
        self.logger = logging.getLogger(f"eremore.{__name__}.{name}")
        self.name = name

    def _export(self, image):
        image = np.clip(image.raw_image, 0, 2**8-1)
        image = image.astype(dtype=np.uint8)
        if len(image.shape) == 3:
            image = image[:, :, ::-1]
        cv2.imwrite(self.path_to_export_image, image)
