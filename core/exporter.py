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
                 input_magnitude: int = 2**14,
                 input_black_level: int = 0,
                 input_white_level: int = 2**12-1,
                 output_black_level: int = 0,
                 output_white_level: int = 2**8-1,
                 path_to_export_image: str = None):
        self.logger = logging.getLogger(f"eremore.{__name__}")
        self.input_magnitude = input_magnitude
        self.input_black_level = input_black_level
        self.input_white_level = input_white_level
        self.output_black_level = output_black_level
        self.output_white_level = output_white_level
        self.path_to_export_image = path_to_export_image
        self._export_mapping_table = self._get_export_mapping_table()

    def update_export_mapping_table(self):
        self._export_mapping_table = self._get_export_mapping_table()

    def export(self, image: Image):
        attributes = get_attributes(self)
        arguments = {'image': image}
        self.logger.debug(f"Exporting with -> attributes: {attributes} | arguments: {arguments}")
        run_and_measure_time(self._export_wrapper, arguments, logger=self.logger)

    def _export_wrapper(self, image: Image):
        image = self._export_mapping_table[image.raw_image]
        self._export(image)

    @abstractmethod
    def _export(self, image):
        pass

    def _get_export_mapping_table(self):
        tone_mapping_table = np.asarray(range(0, self.input_magnitude), dtype=np.float32)
        tone_mapping_table = np.clip(tone_mapping_table, self.input_black_level, self.input_white_level)
        tone_mapping_table -= self.input_black_level
        tone_mapping_table *= (self.output_white_level - self.output_black_level) / (self.input_white_level - self.input_black_level)
        tone_mapping_table += self.output_black_level
        tone_mapping_table = np.clip(tone_mapping_table, 0, 2**8-1)
        return tone_mapping_table.astype(dtype=np.uint8)


class ExporterOpenCV(ExporterBase):
    def __init__(self, name='open_cv'):
        super().__init__()
        self.logger = logging.getLogger(f"eremore.{__name__}.{name}")
        self.name = name

    def _export(self, image):
        if len(image.shape) == 3:
            image = image[:, :, ::-1]
        cv2.imwrite(self.path_to_export_image, image)
