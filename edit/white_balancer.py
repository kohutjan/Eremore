from abc import ABC, abstractmethod

import logging

import numpy as np

from helper.get_attributes import get_attributes
from helper.run_and_measure_time import run_and_measure_time

from core.image import Image


class WhiteBalancer(ABC):
    def __init__(self):
        self.logger = logging.getLogger(f"eremore.{__name__}")

    def white_balance(self, image: Image):
        attributes = get_attributes(self)
        arguments = {'image': image}
        self.logger.debug(f"White balancing with -> attributes: {attributes} | arguments: {arguments}")
        run_and_measure_time(self._white_balance, arguments, logger=self.logger)

    def _white_balance(self, image: Image):
        scales = self._get_color_scales(image)
        red_scale, green_scale, blue_scale = (scales * 3) / np.sum(scales)
        self.logger.debug(f"Estimated color scales - > {red_scale, green_scale, blue_scale}")
        for color_c, color_scale in enumerate((red_scale, green_scale, blue_scale)):
            image.raw_image[:, :, color_c] *= color_scale

    @abstractmethod
    def _get_color_scales(self, image: Image):
        pass


class WhiteBalancerWhitePatch(WhiteBalancer):
    def __init__(self, percentile):
        super().__init__()
        self.logger = logging.getLogger(f"eremore.{__name__}.white_patch")
        self.percentile = percentile

    def _get_color_scales(self, image):
        white = np.percentile(image.raw_image, q=self.percentile, axis=(0, 1))
        scales = 1.0 / white
        return scales


class WhiteBalancerGrayWorld(WhiteBalancer):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(f"eremore.{__name__}.gray_world")

    def _get_color_scales(self, image):
        gray = np.mean(image.raw_image, axis=(0, 1))
        scales = 1.0 / gray
        return scales
