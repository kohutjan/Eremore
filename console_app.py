import argparse
import sys
import os

from core.loader import Loader
from core.demosaicing import Demosaicing


def parseargs():
    print(' '.join(sys.argv))
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-image', required=True, type=str, help="Path to the RAW image file.")
    args = parser.parse_args()
    return args


def main():
    args = parseargs()

    raw_image = Loader.load(args.raw_image)


if __name__ == '__main__':
    main()