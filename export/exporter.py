from abc import ABC, abstractmethod

import logging

import numpy as np
import numpy.typing as npt

from helper.run_and_measure_time import run_and_measure_time


class Exporter(ABC):
    def __init__(self, raw_min: int, raw_max: int, export_min: int, export_max: int):
        self.logger = logging.getLogger(f"eremore.{__name__}")
        self.raw_min = raw_min
        self.raw_max = raw_max
        self.export_min = export_min
        self.export_max = export_max

    def export(self, raw_image: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint8]:
        arguments = {'raw_image': raw_image.shape}
        self.logger.debug(f"Exporting with: -> attributes: {vars(self)} | arguments: {arguments}")
        export_image, elapsed_time = run_and_measure_time(self._export_wrapper,
                                                          {'raw_image': raw_image},
                                                          logger=self.logger)

        return export_image

    def _export_wrapper(self, raw_image: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint8]:
        raw_image = self._export_preprocess(raw_image)
        raw_image = self._export(raw_image)
        return self._export_postprocess(raw_image)

    def _export_preprocess(self, raw_image: npt.NDArray[np.uint16]) -> npt.NDArray[np.float32]:
        raw_image = raw_image.astype(np.float32)
        raw_image = np.clip(raw_image, self.raw_min, self.raw_max)
        return raw_image

    @abstractmethod
    def _export(self, raw_image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        pass

    def _export_postprocess(self, raw_image: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
        raw_image = np.clip(raw_image, self.export_min, self.export_max)
        export_image = raw_image.astype(np.uint8)
        return export_image


class ExporterLinear(Exporter):
    def __init__(self, raw_min=0, raw_max=2**14-1, export_min=0, export_max=255):
        super().__init__(raw_min, raw_max, export_min, export_max)
        self.logger = logging.getLogger(f"eremore.{__name__}.linear")
        self.name = "ExporterLinear"

    def _export(self, raw_image):
        raw_image -= self.raw_min
        scale = (self.export_max - self.export_min) / (self.raw_max - self.raw_min)
        raw_image *= scale
        raw_image += self.export_min
        return raw_image


class ExporterLog(Exporter):
    def __init__(self, raw_min=0, raw_max=2**14-1, export_min=0, export_max=255):
        super().__init__(raw_min, raw_max, export_min, export_max)
        self.logger = logging.getLogger(f"eremore.{__name__}.log")
        self.name = "ExporterLog"

    def _export(self, raw_image):
        raw_image -= self.raw_min + 1
        raw_image = np.log(raw_image)
        raw_max_log = np.log(self.raw_max - self.raw_min + 1)
        scale = (self.export_max - self.export_min) / raw_max_log
        raw_image *= scale
        raw_image += self.export_min
        return raw_image


class ExporterGammaCorrection(Exporter):
    def __init__(self, raw_min=0, raw_max=2**14-1, export_min=0, export_max=255, gamma=1):
        super().__init__(raw_min, raw_max, export_min, export_max)
        self.logger = logging.getLogger(f"eremore.{__name__}.gamma_correction")
        self.name = "ExporterGammaCorrection"
        self.gamma = gamma

    def _export(self, raw_image):
        raw_image -= self.raw_min
        raw_image /= self.raw_max - self.raw_min
        raw_image **= self.gamma
        raw_image *= self.export_max - self.export_min
        raw_image += self.export_min
        return raw_image
