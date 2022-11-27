import logging

import numpy as np
import numpy.typing as npt


class Image:
    def __init__(self, raw_image: npt.NDArray[np.float32], camera_white_balance=None):
        self.logger = logging.getLogger(f"eremore.{__name__}")
        self.raw_image = raw_image
        self.camera_white_balance = camera_white_balance

    def __str__(self):
        return str({'shape': self.raw_image.shape,
                    'type': self.raw_image.dtype})

    def __repr__(self):
        return self.__str__()

