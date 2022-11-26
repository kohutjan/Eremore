from abc import ABC, abstractmethod

import logging

import numpy as np
import numpy.typing as npt

from helper.get_attributes import get_attributes
from helper.run_and_measure_time import run_and_measure_time

from core.image import Image


class ToneMapper(ABC):
    def __init__(self, input_black: int, input_white: int, output_black: int, output_white: int):
        self.logger = logging.getLogger(f"eremore.{__name__}")
        self.input_black = input_black
        self.input_white = input_white
        self.output_black = output_black
        self.output_white = output_white

    def tone_map(self, image: Image):
        attributes = get_attributes(self)
        arguments = {'image': image}
        self.logger.debug(f"Tone mapping with: -> attributes: {attributes} | arguments: {arguments}")
        run_and_measure_time(self._tone_map_wrapper, arguments, logger=self.logger)

    def _tone_map_wrapper(self, image: Image):
        self._tone_map_pre_process(image)
        self._tone_map(image)

    def _tone_map_pre_process(self, image: Image):
        image.raw_image = np.clip(image.raw_image, self.input_black, self.input_white)

    @abstractmethod
    def _tone_map(self, image: Image):
        pass

    def _tone_map_post_process(self, image: Image):
        image.raw_image = np.clip(image.raw_image, self.output_black, self.output_white)


class ToneMapperLinear(ToneMapper):
    def __init__(self, input_black=0, input_white=2**14-1, output_black=0, output_white=255):
        super().__init__(input_black, input_white, output_black, output_white)
        self.logger = logging.getLogger(f"eremore.{__name__}.linear")
        self.name = "ToneMapperLinear"

    def _tone_map(self, image):
        image.raw_image -= self.input_black
        image.raw_image *= (self.output_white - self.output_black) / (self.input_white - self.input_black)
        image.raw_image += self.output_black


class ToneMapperLog(ToneMapper):
    def __init__(self, input_black=0, input_white=2**14-1, output_black=0, output_white=255):
        super().__init__(input_black, input_white, output_black, output_white)
        self.logger = logging.getLogger(f"eremore.{__name__}.log")
        self.name = "ToneMapperLog"

    def _tone_map(self, image):
        image.raw_image -= self.input_black + 1
        image.raw_image = np.log(image.raw_image)
        image.raw_image *= (self.output_white - self.output_black) / np.log(self.input_white - self.input_black + 1)
        image.raw_image += self.output_black


class ToneMapperGammaCorrection(ToneMapper):
    def __init__(self, input_black=0, input_white=2**14-1, output_black=0, output_white=255, gamma=1):
        super().__init__(input_black, input_white, output_black, output_white)
        self.logger = logging.getLogger(f"eremore.{__name__}.gamma_correction")
        self.name = "ToneMapperGammaCorrection"
        self.gamma = gamma

    def _tone_map(self, image):
        image.raw_image -= self.input_black
        image.raw_image /= self.input_white - self.input_black
        image.raw_image **= self.gamma
        image.raw_image *= self.output_white - self.output_black
        image.raw_image += self.output_black
