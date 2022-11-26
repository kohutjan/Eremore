from abc import ABC, abstractmethod

import logging

import numpy as np
import numpy.typing as npt

from helper.get_attributes import get_attributes
from helper.run_and_measure_time import run_and_measure_time

from core.image import Image


class ToneMapper(ABC):
    def __init__(self,
                 input_black_level_correction: int,
                 input_black_level: int, input_white_level: int,
                 output_black_level: int, output_white_level: int,
                 ):
        self.logger = logging.getLogger(f"eremore.{__name__}")
        self.input_black_level_correction = input_black_level_correction
        self.input_black_level = input_black_level
        self.input_white_level = input_white_level
        self.output_black_level = output_black_level
        self.output_white_level = output_white_level

    def tone_map(self, image: Image):
        attributes = get_attributes(self)
        arguments = {'image': image}
        self.logger.debug(f"Tone mapping with: -> attributes: {attributes} | arguments: {arguments}")
        run_and_measure_time(self._tone_map_wrapper, arguments, logger=self.logger)

    def _tone_map_wrapper(self, image: Image):
        self._tone_map_pre_process(image)
        self._tone_map(image)

    def _tone_map_pre_process(self, image: Image):
        image.raw_image -= self.input_black_level_correction
        image.raw_image = np.clip(image.raw_image, self.input_black_level, self.input_white_level)

    @abstractmethod
    def _tone_map(self, image: Image):
        pass

    def _tone_map_post_process(self, image: Image):
        image.raw_image = np.clip(image.raw_image, self.output_black_level, self.output_white_level)


class ToneMapperLinear(ToneMapper):
    def __init__(self,
                 input_black_level_correction,
                 input_black_level, input_white_level,
                 output_black_level, output_white_level):
        super().__init__(input_black_level_correction,
                         input_black_level, input_white_level,
                         output_black_level, output_white_level)
        self.logger = logging.getLogger(f"eremore.{__name__}.linear")
        self.name = "ToneMapperLinear"

    def _tone_map(self, image):
        image.raw_image -= self.input_black_level
        image.raw_image *= (self.output_white_level - self.output_black_level) / (self.input_white_level - self.input_black_level)
        image.raw_image += self.output_black_level


class ToneMapperLog(ToneMapper):
    def __init__(self,
                 input_black_level_correction,
                 input_black_level, input_white_level,
                 output_black_level, output_white_level):
        super().__init__(input_black_level_correction,
                         input_black_level, input_white_level,
                         output_black_level, output_white_level)
        self.logger = logging.getLogger(f"eremore.{__name__}.log")
        self.name = "ToneMapperLog"

    def _tone_map(self, image):
        image.raw_image -= self.input_black_level + 1
        image.raw_image = np.log(image.raw_image)
        image.raw_image *= (self.output_white_level - self.output_black_level) / np.log(self.input_white_level - self.input_black_level + 1)
        image.raw_image += self.output_black_level


class ToneMapperGammaCorrection(ToneMapper):
    def __init__(self,
                 input_black_level_correction,
                 input_black_level, input_white_level,
                 output_black_level, output_white_level,
                 gamma):
        super().__init__(input_black_level_correction,
                         input_black_level, input_white_level,
                         output_black_level, output_white_level)
        self.logger = logging.getLogger(f"eremore.{__name__}.gamma_correction")
        self.name = "ToneMapperGammaCorrection"
        self.gamma = gamma

    def _tone_map(self, image):
        image.raw_image -= self.input_black_level
        image.raw_image /= self.input_white_level - self.input_black_level
        image.raw_image **= self.gamma
        image.raw_image *= self.output_white_level - self.output_black_level
        image.raw_image += self.output_black_level
