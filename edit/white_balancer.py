from abc import ABC, abstractmethod

import logging

import numpy as np

from collections import OrderedDict

from helper.get_attributes import get_attributes
from helper.run_and_measure_time import run_and_measure_time

from core.image import Image


class WhiteBalancer:
    def __init__(self, name: str = 'white_balancer', engine: str = 'none'):
        self.logger = logging.getLogger(f"eremore.{__name__}")
        self.name = name
        self.engine = engine
        self.engines = OrderedDict()
        self.engines['rgb_scale'] = WhiteBalancerRGBScale()
        self.engines['camera'] = WhiteBalancerCamera()
        self.engines['white_patch'] = WhiteBalancerWhitePatch()
        self.engines['gray_world'] = WhiteBalancerGrayWorld()

    def process(self, image: Image):
        if self.engine == 'none':
            return
        if self.engine not in self.engines.keys():
            self.logger.error(f"WhiteBalancer engine {self.engine} does not exists.")
            raise ValueError

        self.engines[self.engine].white_balance(image)

        if self.engine == 'camera' or self.engine == 'white_patch' or self.engine == 'gray_world':
            self.engines['rgb_scale'].r_scale = self.engines[self.engine].r_scale
            self.engines['rgb_scale'].g_scale = self.engines[self.engine].g_scale
            self.engines['rgb_scale'].b_scale = self.engines[self.engine].b_scale


class WhiteBalancerBase(ABC):
    def __init__(self):
        self.logger = logging.getLogger(f"eremore.{__name__}")

    def white_balance(self, image: Image):
        attributes = get_attributes(self)
        arguments = {'image': image}
        self.logger.debug(f"White balancing with -> attributes: {attributes} | arguments: {arguments}")
        run_and_measure_time(self._white_balance, arguments, logger=self.logger)

    @abstractmethod
    def _white_balance(self, image: Image):
        pass


class WhiteBalancerRGBScale(WhiteBalancerBase):
    def __init__(self, name='rgb_scale', r_scale: float = 1, g_scale: float = 1, b_scale: float = 1, normalize=False):
        super().__init__()
        self.logger = logging.getLogger(f"eremore.{__name__}.{name}")
        self.name = name
        self.r_scale = r_scale
        self.g_scale = g_scale
        self.b_scale = b_scale
        self.normalize = normalize

    def set_scales(self, scales):
        self.r_scale = scales[0]
        self.g_scale = scales[1]
        self.b_scale = scales[2]

    def _white_balance(self, image: Image):
        scales = np.asarray([self.r_scale, self.g_scale, self.b_scale])
        if self.normalize:
            scales = (scales * 3) / np.sum(scales)
            self.logger.debug(f"Normalized color scales - > {scales}")
        if np.all(scales == 1):
            return
        image.raw_image *= np.expand_dims(scales, axis=(0, 1))


class WhiteBalancerCamera(WhiteBalancerRGBScale):
    def __init__(self, name='camera'):
        super().__init__(normalize=True)
        self.logger = logging.getLogger(f"eremore.{__name__}.{name}")
        self.name = name

    def _white_balance(self, image: Image):
        if image.camera_white_balance is None or len(image.camera_white_balance) != 3:
            self.logger.warning(f"No camera white balance could be read from the raw image, doing nothing.")
            scales = np.full(3, 1.0)
        else:
            scales = image.camera_white_balance
        self.set_scales(scales)
        super()._white_balance(image)


class WhiteBalancerWhitePatch(WhiteBalancerRGBScale):
    def __init__(self, name='white_patch', percentile: float = 0.97):
        super().__init__(normalize=True)
        self.logger = logging.getLogger(f"eremore.{__name__}.{name}")
        self.name = name
        self.percentile = percentile

    def white_balance(self, image: Image):
        white = np.percentile(image.raw_image, q=self.percentile, axis=(0, 1))
        scales = 1.0 / white
        self.set_scales(scales)
        super()._white_balance(image)


class WhiteBalancerGrayWorld(WhiteBalancerRGBScale):
    def __init__(self, name='gray_world'):
        super().__init__(normalize=True)
        self.logger = logging.getLogger(f"eremore.{__name__}.{name}")
        self.name = name

    def white_balance(self, image: Image):
        gray = np.mean(image.raw_image, axis=(0, 1))
        scales = 1.0 / gray
        self.set_scales(scales)
        super()._white_balance(image)


