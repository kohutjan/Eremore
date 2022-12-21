from abc import ABC, abstractmethod

import logging

import numpy as np

from collections import OrderedDict

from helper.get_attributes import get_attributes
from helper.run_and_measure_time import run_and_measure_time

from core.image import Image


class ToneMapper:
    def __init__(self, name: str = 'tone_mapper', engine: str = None):
        self.logger = logging.getLogger(f"eremore.{__name__}")
        self.name = name
        self.engine = engine
        self.engines = OrderedDict()
        self.engines['linear'] = ToneMapperLinear()
        self.engines['gamma_correction'] = ToneMapperGammaCorrection()

    def process(self, image: Image):
        if self.engine is None:
            return
        if self.engine not in self.engines.keys():
            self.logger.error(f"ToneMapper engine {self.engine} does not exists.")
            raise ValueError

        self.engines[self.engine].tone_map(image)


class ToneMapperBase(ABC):
    def __init__(self,
                 input_magnitude: int = 2**14,
                 input_black_level_correction: int = 512,
                 input_black_level: int = 0, input_white_level: int = 2**12-1):
        self.logger = logging.getLogger(f"eremore.{__name__}")
        self.input_magnitude = input_magnitude
        self.input_black_level_correction = input_black_level_correction
        self.input_black_level = input_black_level
        self.input_white_level = input_white_level
        self._tone_mapping_table = None

    def tone_map(self, image: Image):
        attributes = get_attributes(self)
        arguments = {'image': image}
        self.logger.debug(f"Tone mapping with -> attributes: {attributes} | arguments: {arguments}")
        run_and_measure_time(self._tone_map_wrapper, arguments, logger=self.logger)

    def _tone_map_wrapper(self, image: Image):
        image.raw_image = self._tone_mapping_table[image.raw_image]
        self._tone_map_camera_white_balance(image)

    @abstractmethod
    def _tone_map(self, tone_mapping_table):
        pass

    def _get_tone_mapping_table(self):
        tone_mapping_table = np.asarray(range(0, self.input_magnitude), dtype=np.float32)
        tone_mapping_table -= self.input_black_level_correction
        tone_mapping_table = np.clip(tone_mapping_table, self.input_black_level, self.input_white_level)
        tone_mapping_table = self._tone_map(tone_mapping_table)
        return tone_mapping_table.astype(dtype=np.uint16)

    def _tone_map_camera_white_balance(self, image: Image):
        if image.camera_white_balance is None or len(image.camera_white_balance) != 3:
            return
        image.camera_white_balance -= self.input_black_level_correction
        self._tone_map(image.camera_white_balance)

    def _update_tone_mapping_table(self):
        self._tone_mapping_table = self._get_tone_mapping_table()

    def set(self, input_magnitude=None,
            input_black_level_correction=None,
            input_black_level=None, input_white_level=None):
        self._set(input_magnitude,
                  input_black_level_correction,
                  input_black_level, input_white_level)
        self._update_tone_mapping_table()

    def _set(self, input_magnitude=None,
             input_black_level_correction=None,
             input_black_level=None, input_white_level=None):
        if input_magnitude is not None:
            self.input_magnitude = input_magnitude
        if input_black_level_correction is not None:
            self.input_black_level_correction = input_black_level_correction
        if input_black_level is not None:
            self.input_black_level = input_black_level
        if input_white_level is not None:
            self.input_white_level = input_white_level


class ToneMapperLinear(ToneMapperBase):
    def __init__(self, name='linear'):
        super().__init__()
        self.logger = logging.getLogger(f"eremore.{__name__}.{name}")
        self.name = name

    def _tone_map(self, tone_mapping_table):
        tone_mapping_table -= self.input_black_level
        return tone_mapping_table


class ToneMapperGammaCorrection(ToneMapperBase):
    def __init__(self, name='gamma_correction', gamma: float = 1.0):
        super().__init__()
        self.logger = logging.getLogger(f"eremore.{__name__}.{name}")
        self.name = name
        self.gamma = gamma

    def _tone_map(self, tone_mapping_table):
        tone_mapping_table -= self.input_black_level
        tone_mapping_table /= self.input_white_level - self.input_black_level
        tone_mapping_table **= self.gamma
        tone_mapping_table *= self.input_white_level - self.input_black_level
        return tone_mapping_table

    def set(self, input_magnitude=None,
            input_black_level_correction=None,
            input_black_level=None, input_white_level=None,
            gamma=None):
        super()._set(input_magnitude,
                     input_black_level_correction,
                     input_black_level, input_white_level)
        if gamma is not None:
            self.gamma = gamma
        self._update_tone_mapping_table()



