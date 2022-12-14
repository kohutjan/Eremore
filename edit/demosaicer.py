from abc import ABC, abstractmethod

import logging

import scipy
import numpy as np

from typing import Tuple
from collections import OrderedDict

from helper.get_attributes import get_attributes
from helper.run_and_measure_time import run_and_measure_time

from core.image import Image


class Demosaicer:
    def __init__(self, name: str = 'demosaicer', engine: str = None):
        self.logger = logging.getLogger(f"eremore.{__name__}")
        self.name = name
        self.engine = engine
        self.engines = OrderedDict()
        self.engines['bayer_splitter'] = BayerSplitter()
        self.engines['copy'] = DemosaicerCopy()
        self.engines['linear'] = DemosaicerLinear()

    def process(self, image: Image):
        if self.engine is None:
            return
        if self.engine not in self.engines.keys():
            self.logger.error(f"Demosaicer engine {self.engine} does not exists.")
            raise ValueError

        self.engines[self.engine].demosaice(image)


class DemosaicerBase(ABC):
    def __init__(self, blue_loc: Tuple[int, int] = (1, 1)):
        self.logger = logging.getLogger(f"eremore.{__name__}")
        self.name = None
        self.blue_loc = blue_loc
        if self.blue_loc == (0, 0):
            self._red_loc = (1, 1)
            self._green_x_loc = (1, 0)
        elif self.blue_loc == (0, 1):
            self._red_loc = (1, 0)
            self._green_x_loc = (0, 1)
        elif self.blue_loc == (1, 0):
            self._red_loc = (0, 1)
            self._green_x_loc = (0, 1)
        elif self.blue_loc == (1, 1):
            self._red_loc = (0, 0)
            self._green_x_loc = (1, 0)
        else:
            self.logger.error(f"Wrong value of blue_loc: {blue_loc}")
            raise ValueError

    def demosaice(self, image: Image):
        attributes = get_attributes(self)
        arguments = {'image': image}
        self.logger.debug(f"Demosaicing with -> attributes: {attributes} | arguments: {arguments}")
        run_and_measure_time(self._demosaice, arguments, logger=self.logger)

    @abstractmethod
    def _demosaice(self, image: Image):
        pass

    def set(self, name=None, blue_loc=None):
        self._set(name, blue_loc)

    def _set(self, name=None, blue_loc=None):
        if name is not None:
            self.name = name
            self.logger = logging.getLogger(f"eremore.{__name__}.{name}")
        if blue_loc is not None:
            self.blue_loc = blue_loc


class BayerSplitter(DemosaicerBase):
    def __init__(self):
        super().__init__()

    def _demosaice(self, image: Image):
        input_raw_image = image.raw_image
        height, width = input_raw_image.shape
        out_raw_image = np.zeros((height, width, 3), dtype=np.uint16)

        for color_loc, color_c in zip((self._red_loc, self.blue_loc), (0, 2)):
            out_raw_image[color_loc[0]::2, color_loc[1]::2, color_c] = input_raw_image[color_loc[0]::2,
                                                                       color_loc[1]::2]
        for i in range(2):
            out_raw_image[i::2, self._green_x_loc[i]::2, 1] = input_raw_image[i::2,
                                                              self._green_x_loc[i]::2]

        image.raw_image = out_raw_image


class DemosaicerCopy(BayerSplitter):
    def __init__(self, name='copy'):
        super().__init__()
        self.logger = logging.getLogger(f"eremore.{__name__}.{name}")
        self.name = name

    def _demosaice(self, image: Image):
        super()._demosaice(image)
        for color_loc, color_c in zip((self._red_loc, self.blue_loc), (0, 2)):
            image.raw_image[color_loc[0]::2, abs(color_loc[1] - 1)::2, color_c] = image.raw_image[color_loc[0]::2,
                                                                                                  color_loc[1]::2,
                                                                                                  color_c]
            image.raw_image[abs(color_loc[0] - 1)::2, :, color_c] = image.raw_image[color_loc[0]::2,
                                                                                    :,
                                                                                    color_c]
        for green_y_loc, green_x_loc in enumerate(self._green_x_loc):
            image.raw_image[green_y_loc::2, abs(green_x_loc - 1)::2, 1] = image.raw_image[green_y_loc::2,
                                                                                          green_x_loc::2,
                                                                                          1]


class DemosaicerLinear(BayerSplitter):
    def __init__(self, name='linear'):
        super().__init__()
        self.logger = logging.getLogger(f"eremore.{__name__}.{name}")
        self.name = name

    def _demosaice(self, image: Image):
        super()._demosaice(image)
        image.raw_image = image.raw_image.astype(dtype=np.float32)
        red_blue_kernel_2 = np.array([[0.5, 0.5]], dtype=np.float32)
        red_blue_kernel_4 = np.array([[0.25, 0.25],
                                      [0.25, 0.25]], dtype=np.float32)
        green_kernel = np.array([[0.0, 0.25, 0.0],
                                 [0.25, 0.0, 0.25],
                                 [0.0, 0.25, 0.0]], dtype=np.float32)

        for color_loc, color_c in zip((self._red_loc, self.blue_loc), (0, 2)):
            color_src = image.raw_image[color_loc[0]::2, color_loc[1]::2, color_c]
            image.raw_image[color_loc[0]::2, abs(color_loc[1] - 1)::2, color_c] = \
                scipy.signal.convolve2d(color_src, red_blue_kernel_2, mode='same')
            image.raw_image[abs(color_loc[0] - 1)::2, color_loc[1]::2, color_c] = \
                scipy.signal.convolve2d(color_src, np.transpose(red_blue_kernel_2), mode='same')
            image.raw_image[abs(color_loc[0] - 1)::2, abs(color_loc[1] - 1)::2, color_c] = \
                scipy.signal.convolve2d(color_src, red_blue_kernel_4, mode='same')

        green_out = scipy.signal.convolve2d(image.raw_image[:, :, 1], green_kernel, mode='same')
        for green_y_loc, green_x_loc in enumerate(self._green_x_loc):
            image.raw_image[green_y_loc::2, abs(green_x_loc - 1)::2, 1] = green_out[green_y_loc::2,
                                                                                    abs(green_x_loc - 1)::2]
        image.raw_image = image.raw_image.astype(dtype=np.uint16)
