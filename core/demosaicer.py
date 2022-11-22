from abc import ABC, abstractmethod

import logging

import scipy
import numpy as np
import numpy.typing as npt
from typing import Tuple

from timeit import default_timer as timer

logger = logging.getLogger(f"eremore.{__name__}")


class Demosaicer(ABC):
    def __init__(self, blue_loc: Tuple[int, int]):
        self.blue_loc = blue_loc
        if self.blue_loc == (0, 0):
            self.red_loc = (1, 1)
            self.green_x_loc = (1, 0)
        elif self.blue_loc == (0, 1):
            self.red_loc = (1, 0)
            self.green_x_loc = (0, 1)
        elif self.blue_loc == (1, 0):
            self.red_loc = (0, 1)
            self.green_x_loc = (0, 1)
        elif self.blue_loc == (1, 1):
            self.red_loc = (0, 0)
            self.green_x_loc = (1, 0)
        else:
            logger.error(f"Wrong value of blue_loc: {blue_loc}")
            raise ValueError

    def demosaice(self, raw_image: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint8]:
        arguments = {'raw_image': raw_image.shape}
        logger.debug(f"Demosaicing with: -> attributes: {vars(self)} | arguments: {arguments}")
        start = timer()
        export_image = self._demosaice(raw_image)
        end = timer()
        elapsed_time = end - start
        logger.debug(f"Elapsed time: {elapsed_time}")
        return export_image

    def _demosaice(self, raw_image: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint8]:
        height, width = raw_image.shape
        out_raw_image = np.zeros((height, width, 3))

        out_raw_image[self.red_loc[0]::2, self.red_loc[1]::2, 0] = raw_image[self.red_loc[0]::2, self.red_loc[1]::2]
        for i in range(2):
            out_raw_image[i::2, self.green_x_loc[i]::2, 1] = raw_image[i::2, self.green_x_loc[i]::2]
        out_raw_image[self.blue_loc[0]::2, self.blue_loc[1]::2, 2] = raw_image[self.blue_loc[0]::2, self.blue_loc[1]::2]

        return out_raw_image


class BayerSplitter(Demosaicer):
    def __init__(self, blue_loc):
        super().__init__(blue_loc)
        self.name = "BayerSplitter"

    def _demosaice(self, raw_image):
        return super()._demosaice(raw_image)


class LinearDemosaicer(Demosaicer):
    def __init__(self, blue_loc):
        super().__init__(blue_loc)
        self.name = "LinearDemosaicer"

    def _demosaice(self, raw_image):
        raw_image = super()._demosaice(raw_image)
        #raw_image.astype(np.float32)
        #red_kernel = np.array([[0.5, 0, 0.5]])
        #green_kernel = np.array([[0.0, 0.5, 0.0],
        #                         [0.5, 0.0, 0.5],
        #                         [0.0, 0.5, 0.0]])
        #blue_kernel = red_kernel
        #raw_image[:, :, 0] = scipy.signal.convolve2d(raw_image[:, :, 0], kernel=red_kernel)
        #raw_image[:, :, 1] = scipy.signal.convolve2d(raw_image[:, :, 1], kernel=green_kernel)
        #raw_image[:, :, 2] = scipy.signal.convolve2d(raw_image[:, :, 2], kernel=blue_kernel)
        return raw_image

