import cv2

import logging
import argparse
import sys
import os

from core.loader import LoaderSimplePyRaw
from edit.rotator import RotatorBasic
from export.exporter import ExporterLinear, ExporterLog, ExporterGammaCorrection
from core.demosaicer import BayerSplitter

logger = logging.getLogger(f"eremore.{__name__}")


def parseargs():
    print(' '.join(sys.argv))
    parser = argparse.ArgumentParser()
    # Input
    parser.add_argument('--raw-image', required=True, type=str, help="Path to the RAW image.")

    # Demosaicer
    parser.add_argument('--blue-loc', type=str)

    # Exporter
    parser.add_argument('--exporter', default='log', choices=['linear', 'log', 'gamma_correction'])
    parser.add_argument('--raw-min', default=0, type=int)
    parser.add_argument('--raw-max', default=2**14-1, type=int)
    parser.add_argument('--export-min', default=0, type=int)
    parser.add_argument('--export-max', default=255, type=int)
    # Exporter gamma_correction
    parser.add_argument('--gamma', default=1, type=float)

    # Rotator
    parser.add_argument('--rotate-k', type=int)

    # Output
    parser.add_argument('--export-image', required=True, type=str, help="Path to save the exported image.")

    parser.add_argument('--logging-level', default=logging.INFO)

    args = parser.parse_args()
    return args


def main():
    args = parseargs()
    logging.basicConfig(format='%(name)s %(asctime)s %(levelname)-8s %(message)s', level=args.logging_level,
                        datefmt='%Y-%m-%d %H:%M:%S')

    loader = LoaderSimplePyRaw()
    raw_image = loader.load(args.raw_image)

    if args.blue_loc is not None:
        blue_loc = [int(x) for x in args.blue_loc.split(",")]
        blue_loc = (blue_loc[0], blue_loc[1])
        bayer_splitter = BayerSplitter(blue_loc)
        raw_image = bayer_splitter.demosaice(raw_image)

    if args.exporter == 'linear':
        exporter = ExporterLinear(raw_min=args.raw_min, raw_max=args.raw_max,
                                  export_min=args.export_min, export_max=args.export_max)
    elif args.exporter == 'log':
        exporter = ExporterLog(raw_min=args.raw_min, raw_max=args.raw_max,
                               export_min=args.export_min, export_max=args.export_max)
    elif args.exporter == 'gamma_correction':
        exporter = ExporterGammaCorrection(raw_min=args.raw_min, raw_max=args.raw_max,
                                           export_min=args.export_min, export_max=args.export_max,
                                           gamma=args.gamma)
    else:
        logger.error(f"Exporter {args.exporter} does not exists.")
        raise ValueError

    export_image = exporter.export(raw_image)

    if args.rotate_k is not None:
        rotator = RotatorBasic()
        export_image = rotator.rotate_k(export_image, k=args.rotate_k)

    if len(export_image.shape) == 3:
        export_image = export_image[:, :, ::-1]
    cv2.imwrite(args.export_image, export_image)


if __name__ == '__main__':
    main()
