from abc import ABC, abstractmethod

import logging

import numpy as np
import numpy.typing as npt

from timeit import default_timer as timer

logger = logging.getLogger(f"eremore.{__name__}")


class Exporter(ABC):
    def export(self, raw_image: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint8]:
        arguments = {'raw_image': raw_image.shape}
        logger.debug(f"Exporting with: -> attributes: {vars(self)} | arguments: {arguments}")
        start = timer()
        export_image = self._export(raw_image)
        end = timer()
        elapsed_time = end - start
        logger.debug(f"Elapsed time: {elapsed_time}")
        return export_image

    @abstractmethod
    def _export(self, raw_image: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint8]:
        pass


class LinearExporter(Exporter):
    def __init__(self, raw_min=None, raw_max=None, export_min=0, export_max=255):
        self.name = "LinearExporter"
        self.raw_min = raw_min
        self.raw_max = raw_max
        self.export_min = export_min
        self.export_max = export_max

    def _export(self, raw_image):
        raw_image = raw_image.astype(np.float32)
        raw_min = self.raw_min
        raw_max = self.raw_max
        export_min = self.export_min
        export_max = self.export_max

        if raw_min is None:
            raw_min = np.min(raw_image)
        if raw_max is None:
            raw_max = np.max(raw_image)

        raw_image = np.clip(raw_image, raw_min, raw_max)

        raw_image -= raw_min
        raw_image /= raw_max - raw_min
        raw_image *= export_max - export_min
        raw_image += export_min

        raw_image = np.clip(raw_image, 0, 255)

        return raw_image.astype(np.uint8)
