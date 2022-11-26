from abc import ABC, abstractmethod

import logging

import numpy as np
import cv2

from helper.get_attributes import get_attributes
from helper.run_and_measure_time import run_and_measure_time

from core.image import Image


class Exporter(ABC):
    def __init__(self):
        self.logger = logging.getLogger(f"eremore.{__name__}")

    def export(self, image: Image, path_to_export_image: str):
        attributes = get_attributes(self)
        arguments = {'image': image, 'path_to_export_image': path_to_export_image}
        self.logger.debug(f"Exporting with: -> attributes: {attributes} | arguments: {arguments}")
        run_and_measure_time(self._export, arguments, logger=self.logger)

    @abstractmethod
    def _export(self, image: Image, path_to_export_image: str):
        pass


class ExporterOpenCV(Exporter):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(f"eremore.{__name__}.open_cv")
        self.name = "ExporterOpenCV"

    def _export(self, image, path_to_export_image):
        export_image = np.clip(image.raw_image, 0, 255)
        export_image = export_image.astype(np.uint8)
        if len(export_image.shape) == 3:
            export_image = export_image[:, :, ::-1]
        cv2.imwrite(path_to_export_image, export_image)
