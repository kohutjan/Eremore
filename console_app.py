import logging
import argparse
import sys
import os

from core.loader import SimpleRawPyLoader
from core.demosaicing import Demosaicing

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
module_logger = logging.getLogger('eremore.console_app')


def parseargs():
    print(' '.join(sys.argv))
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-image', required=True, type=str, help="Path to the RAW image file.")
    args = parser.parse_args()
    return args


def main():
    args = parseargs()

    simple_rawpy_loader = SimpleRawPyLoader()
    raw_image = simple_rawpy_loader.load(args.raw_image)
    logging.info(f"INPUT SHAPE {raw_image.shape}")


if __name__ == '__main__':
    main()
