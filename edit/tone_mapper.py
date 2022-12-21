from abc import ABC, abstractmethod

import logging

import numpy as np

from collections import OrderedDict

from helper.get_attributes import get_attributes
from helper.run_and_measure_time import run_and_measure_time

from core.image import Image


class ToneMapper:
    def __init__(self, name: str = 'tone_mapper', engine: str = 'none'):
        self.logger = logging.getLogger(f"eremore.{__name__}")
        self.name = name
        self.engine = engine
        self.engines = OrderedDict()
        self.engines['linear'] = ToneMapperLinear()
        self.engines['log'] = ToneMapperLog()
        self.engines['gamma_correction'] = ToneMapperGammaCorrection()

    def process(self, image: Image):
        if self.engine == 'none':
            return
        if self.engine not in self.engines.keys():
            self.logger.error(f"ToneMapper engine {self.engine} does not exists.")
            raise ValueError

        self.engines[self.engine].tone_map(image)


class ToneMapperBase(ABC):
    def __init__(self,
                 input_black_level_correction: int = 512,
                 input_black_level: int = 0, input_white_level: int = 2**12-1,
                 output_black_level: int = 0, output_white_level: int = 2**8-1):
        self.logger = logging.getLogger(f"eremore.{__name__}")
        self.input_black_level_correction = input_black_level_correction
        self.input_black_level = input_black_level
        self.input_white_level = input_white_level
        self.output_black_level = output_black_level
        self.output_white_level = output_white_level

    def tone_map(self, image: Image):
        attributes = get_attributes(self)
        arguments = {'image': image}
        self.logger.debug(f"Tone mapping with -> attributes: {attributes} | arguments: {arguments}")
        run_and_measure_time(self._tone_map_wrapper, arguments, logger=self.logger)

    def _tone_map_wrapper(self, image: Image):
        self._tone_map_pre_process(image)
        self._tone_map(image.raw_image)
        self._tone_map_post_process(image)
        self._tone_map_camera_white_balance(image)

    def _tone_map_pre_process(self, image: Image):
        image.raw_image -= self.input_black_level_correction
        image.raw_image = np.clip(image.raw_image, self.input_black_level, self.input_white_level)

    @abstractmethod
    def _tone_map(self, image):
        pass

    def _tone_map_post_process(self, image: Image):
        image.raw_image = np.clip(image.raw_image, self.output_black_level, self.output_white_level)

    def _tone_map_camera_white_balance(self, image: Image):
        if image.camera_white_balance is None or len(image.camera_white_balance) != 3:
            return
        image.camera_white_balance -= self.input_black_level_correction
        self._tone_map(image=image.camera_white_balance)


class ToneMapperLinear(ToneMapperBase):
    def __init__(self, name='linear'):
        super().__init__()
        self.logger = logging.getLogger(f"eremore.{__name__}.{name}")
        self.name = name

    def _tone_map(self, image):
        image -= self.input_black_level
        image *= (self.output_white_level - self.output_black_level) / (
                    self.input_white_level - self.input_black_level)
        image += self.output_black_level


class ToneMapperLog(ToneMapperBase):
    def __init__(self, name='log'):
        super().__init__()
        self.logger = logging.getLogger(f"eremore.{__name__}.{name}")
        self.name = name

    def _tone_map(self, image):
        image -= self.input_black_level + 1
        image = np.log(image)
        image *= (self.output_white_level - self.output_black_level) / np.log(
            self.input_white_level - self.input_black_level + 1)
        image += self.output_black_level


class ToneMapperGammaCorrection(ToneMapperBase):
    def __init__(self, name='gamma_correction', gamma: float = 1.0):
        super().__init__()
        self.logger = logging.getLogger(f"eremore.{__name__}.{name}")
        self.name = name
        self.gamma = gamma

    def _tone_map(self, image):
        image -= self.input_black_level
        image /= self.input_white_level - self.input_black_level
        image **= self.gamma
        image *= self.output_white_level - self.output_black_level
        image += self.output_black_level


