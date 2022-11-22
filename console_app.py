import cv2

import logging
import argparse
import sys
import os


from core.loader import SimpleRawPyLoader
from export.exporter import LinearExporter
from core.demosaicing import Demosaicing

logger = logging.getLogger(f"eremore.{__name__}")

def parseargs():
    print(' '.join(sys.argv))
    parser = argparse.ArgumentParser()
    # Input
    parser.add_argument('--raw-image', required=True, type=str, help="Path to the RAW image.")

    # Exporter
    parser.add_argument('--raw-min', type=int)
    parser.add_argument('--raw-max', type=int)
    parser.add_argument('--export-min', default=0, type=int)
    parser.add_argument('--export-max', default=255, type=int)

    # Output
    parser.add_argument('--export-image', required=True, type=str, help="Path to save the exported image.")

    parser.add_argument('--logging-level', default=logging.INFO)

    args = parser.parse_args()
    return args


def main():
    args = parseargs()
    logging.basicConfig(format='%(name)s %(asctime)s %(levelname)-8s %(message)s', level=args.logging_level,
                        datefmt='%Y-%m-%d %H:%M:%S')

    simple_rawpy_loader = SimpleRawPyLoader()
    raw_image = simple_rawpy_loader.load(args.raw_image)
    logger.info(f"RAW image shape: {raw_image.shape}")
    linear_exporter = LinearExporter(raw_min=args.raw_min, raw_max=args.raw_max,
                                     export_min=args.export_min, export_max=args.export_max)
    export_image = linear_exporter.export(raw_image)
    logger.info(f"Exported image shape: {export_image.shape}")
    cv2.imwrite(args.export_image, export_image)


if __name__ == '__main__':
    main()
