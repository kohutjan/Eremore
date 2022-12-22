from abc import ABC, abstractmethod

import logging

import numpy as np

from collections import OrderedDict

from helper.get_attributes import get_attributes
from helper.run_and_measure_time import run_and_measure_time

from core.image import Image


class WhiteBalancer:
    def __init__(self, name: str = 'white_balancer', engine: str = None):
        self.logger = logging.getLogger(f"eremore.{__name__}")
        self.name = name
        self.engine = engine
        self.engines = OrderedDict()
        self.engines['rgb'] = WhiteBalancerRGB()
        self.engines['camera'] = WhiteBalancerCamera()
        self.engines['white_patch'] = WhiteBalancerWhitePatch()
        self.engines['gray_world'] = WhiteBalancerGrayWorld()

    def process(self, image: Image):
        if self.engine is None:
            return
        if self.engine not in self.engines.keys():
            self.logger.error(f"WhiteBalancer engine {self.engine} does not exists.")
            raise ValueError

        self.engines[self.engine].white_balance(image)


class WhiteBalancerBase(ABC):
    def __init__(self,
                 input_magnitude: int = 2**14,
                 input_black_level: int = 0, input_white_level: int = 2**12-1):
        self.name = None
        self.logger = logging.getLogger(f"eremore.{__name__}")
        self.input_magnitude = input_magnitude
        self.input_black_level = input_black_level
        self.input_white_level = input_white_level
        self._white_balance_mapping_table = None

    def white_balance(self, image: Image):
        attributes = get_attributes(self)
        arguments = {'image': image}
        self.logger.debug(f"White balancing with -> attributes: {attributes} | arguments: {arguments}")
        run_and_measure_time(self._white_balance_wrapper, arguments, logger=self.logger)

    def _white_balance_wrapper(self, image: Image):
        if self._white_balance_mapping_table is None:
            white_balance_mapping_table = self._get_white_balance_mapping_table(image)
        else:
            white_balance_mapping_table = self._white_balance_mapping_table
        image.raw_image = np.stack([white_balance_mapping_table[0, image.raw_image[:, :, 0]],
                                    white_balance_mapping_table[1, image.raw_image[:, :, 1]],
                                    white_balance_mapping_table[2, image.raw_image[:, :, 2]]], axis=-1)

    @abstractmethod
    def _white_balance(self, white_balance_mapping_table, image: Image = None):
        pass

    def _get_white_balance_mapping_table(self, image: Image = None):
        white_balance_mapping_table = np.asarray([range(0, self.input_magnitude),
                                                  range(0, self.input_magnitude),
                                                  range(0, self.input_magnitude)], dtype=np.float32)
        white_balance_mapping_table = np.clip(white_balance_mapping_table, self.input_black_level, self.input_white_level)
        white_balance_mapping_table = self._white_balance(white_balance_mapping_table, image)
        return white_balance_mapping_table.astype(dtype=np.uint16)

    def set(self,
            name=None,
            input_magnitude=None,
            input_black_level=None, input_white_level=None):
        self._set(name,
                  input_magnitude,
                  input_black_level, input_white_level)

    def _set(self,
             name=None,
             input_magnitude=None,
             input_black_level=None, input_white_level=None):
        if name is not None:
            self.name = name
            self.logger = logging.getLogger(f"eremore.{__name__}.{name}")
        if input_magnitude is not None:
            self.input_magnitude = input_magnitude
        if input_black_level is not None:
            self.input_black_level = input_black_level
        if input_white_level is not None:
            self.input_white_level = input_white_level


class WhiteBalancerRGB(WhiteBalancerBase):
    def __init__(self, name='rgb_scale', r_scale: float = 1, g_scale: float = 1, b_scale: float = 1):
        super().__init__()
        self.logger = logging.getLogger(f"eremore.{__name__}.{name}")
        self.name = name
        self.r_scale = r_scale
        self.g_scale = g_scale
        self.b_scale = b_scale

    def _white_balance(self, white_balance_mapping_table, image: Image = None):
        scales = np.asarray([self.r_scale, self.g_scale, self.b_scale])
        return RGBScale.scale(white_balance_mapping_table, scales)

    def set(self,
            name=None,
            input_magnitude=None,
            input_black_level=None, input_white_level=None,
            r_scale=None, g_scale=None, b_scale=None):
        super()._set(name,
                     input_magnitude,
                     input_black_level, input_white_level)
        if r_scale is not None:
            self.r_scale = r_scale
        if g_scale is not None:
            self.g_scale = g_scale
        if b_scale is not None:
            self.b_scale = b_scale
        self._white_balance_mapping_table = self._get_white_balance_mapping_table(image=None)


class WhiteBalancerCamera(WhiteBalancerBase):
    def __init__(self, name='camera'):
        super().__init__()
        self.logger = logging.getLogger(f"eremore.{__name__}.{name}")
        self.name = name

    def _white_balance(self, white_balance_mapping_table, image: Image):
        if image.camera_white_balance is None or len(image.camera_white_balance) != 3:
            self.logger.warning(f"No camera white balance could be read from the raw image, doing nothing.")
            scales = np.full(3, 1.0)
        else:
            scales = image.camera_white_balance
        return RGBScale.scale(white_balance_mapping_table, scales, normalize=True)


class WhiteBalancerWhitePatch(WhiteBalancerBase):
    def __init__(self, name='white_patch', percentile: float = 0.97):
        super().__init__()
        self.logger = logging.getLogger(f"eremore.{__name__}.{name}")
        self.name = name
        self.percentile = percentile

    def _white_balance(self, white_balance_mapping_table, image: Image):
        white = np.percentile(image.raw_image, q=self.percentile, axis=(0, 1))
        scales = 1.0 / white
        return RGBScale.scale(white_balance_mapping_table, scales, normalize=True)

    def set(self, name=None,
            input_magnitude=None,
            input_black_level=None, input_white_level=None,
            percentile=None):
        super()._set(name,
                     input_magnitude,
                     input_black_level, input_white_level)
        if percentile is not None:
            self.percentile = percentile


class WhiteBalancerGrayWorld(WhiteBalancerBase):
    def __init__(self, name='gray_world'):
        super().__init__()
        self.logger = logging.getLogger(f"eremore.{__name__}.{name}")
        self.name = name

    def _white_balance(self, white_balance_mapping_table, image: Image):
        gray = np.mean(image.raw_image, axis=(0, 1))
        scales = 1.0 / gray
        return RGBScale.scale(white_balance_mapping_table, scales, normalize=True)


class RGBScale:
    logger = logging.getLogger(f"eremore.{__name__}.rgb_scale")

    @staticmethod
    def scale(white_balance_mapping_table, scales, normalize=False):
        if normalize:
            scales = (scales * 3) / np.sum(scales)
            RGBScale.logger.debug(f"Normalized color scales - > {scales}")
        white_balance_mapping_table *= np.expand_dims(scales, axis=1)
        return white_balance_mapping_table
