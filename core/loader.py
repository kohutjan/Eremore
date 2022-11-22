from abc import ABC, abstractmethod


import rawpy
import numpy as np
import numpy.typing as npt


class Loader(ABC):
    @abstractmethod
    def load(self, path_to_raw_image: str) -> npt.NDArray[np.uint16]:
        pass


class SimpleRawPyLoader(Loader):
    def __init__(self):
        pass

    def load(self, path_to_raw_image):
        with rawpy.imread(path_to_raw_image) as rawpy_loader:
            raw_image = rawpy_loader.raw_image.copy()
        return raw_image
