import cv2

import logging
import argparse
import sys
import json
import pprint

from core.loader import LoaderRawPy
from core.demosaicer import BayerSplitter, DemosaicerCopy, DemosaicerLinear
from core.tone_mapper import ToneMapperLinear, ToneMapperLog, ToneMapperGammaCorrection
from edit.white_balancer import WhiteBalancerCamera, WhiteBalancerWhitePatch, WhiteBalancerGrayWorld
from edit.rotator import RotatorNumPy90
from core.exporter import ExporterOpenCV

logger = logging.getLogger(f"eremore.{__name__}")


def parseargs():
    print(' '.join(sys.argv))
    parser = argparse.ArgumentParser()

    group_loader = parser.add_argument_group('Loader')
    group_loader.add_argument('--loader', default='rawpy', choices=['rawpy'])
    group_loader.add_argument('--path-to-raw-image', required=True, type=str, help="Path to the RAW image.")

    group_tone_mapper = parser.add_argument_group('ToneMapper')
    group_tone_mapper.add_argument('--tone-mapper', choices=['linear', 'log', 'gamma_correction'])
    group_tone_mapper.add_argument('--input-black-level-correction', default=0, type=int)
    group_tone_mapper.add_argument('--input-black-level', default=0, type=int)
    group_tone_mapper.add_argument('--input-white-level', default=2**14-1, type=int)
    group_tone_mapper.add_argument('--output-black-level', default=0, type=int)
    group_tone_mapper.add_argument('--output-white-level', default=255, type=int)
    group_tone_mapper_gamma_correction = parser.add_argument_group('ToneMapperGammaCorrection')
    group_tone_mapper_gamma_correction.add_argument('--gamma', default=1, type=float)

    group_demosaicer = parser.add_argument_group('Demosaicer')
    group_demosaicer.add_argument('--demosaicer', choices=['bayer_splitter', 'copy', 'linear'])
    group_demosaicer.add_argument('--blue-loc', default='1,1', choices=['00', '01', '10', '11'])

    group_white_balancer = parser.add_argument_group('WhiteBalancer')
    group_white_balancer.add_argument('--white-balancer', choices=['camera', 'white_patch', 'gray_world'])
    group_white_balancer_white_patch = parser.add_argument_group('WhiteBalancerWhitePatch')
    group_white_balancer_white_patch.add_argument('--percentile', default=97, type=float)

    group_rotator = parser.add_argument_group('Rotator')
    group_rotator.add_argument('--rotator', choices=['numpy90'])
    group_rotator.add_argument('--k', type=int)

    group_exporter = parser.add_argument_group('Exporter')
    group_exporter.add_argument('--exporter', default='open_cv', choices=['open_cv'])
    group_exporter.add_argument('--path-to-export-image', required=True, type=str, help="Path to save the exported image.")

    parser.add_argument('--logging-level', default=logging.INFO)

    args = parser.parse_args()
    return args


def main():
    args = parseargs()
    #logging.basicConfig(format='%(name)s %(asctime)s %(levelname)-8s %(message)s', level=args.logging_level,
    #                    datefmt='%Y-%m-%d %H:%M:%S')
    logging.basicConfig(format='%(name)s %(levelname)-8s %(message)s', level=args.logging_level)

    # Loader
    # ##################################################################################################################
    if args.loader == 'rawpy':
        loader = LoaderRawPy()
    else:
        logger.error(f"Loader {args.loader} does not exist.")
        raise ValueError

    image = loader.load(path_to_raw_image=args.path_to_raw_image)
    # ##################################################################################################################

    # ToneMapper
    # ##################################################################################################################
    if args.tone_mapper is not None:
        if args.tone_mapper == 'linear':
            tone_mapper = ToneMapperLinear(input_black_level_correction=args.input_black_level_correction,
                                           input_black_level=args.input_black_level,
                                           input_white_level=args.input_white_level,
                                           output_black_level=args.output_black_level,
                                           output_white_level=args.output_white_level)
        elif args.tone_mapper == 'log':
            tone_mapper = ToneMapperLog(input_black_level_correction=args.input_black_level_correction,
                                        input_black_level=args.input_black_level,
                                        input_white_level=args.input_white_level,
                                        output_black_level=args.output_black_level,
                                        output_white_level=args.output_white_level)
        elif args.tone_mapper == 'gamma_correction':
            tone_mapper = ToneMapperGammaCorrection(input_black_level_correction=args.input_black_level_correction,
                                                    input_black_level=args.input_black_level,
                                                    input_white_level=args.input_white_level,
                                                    output_black_level=args.output_black_level,
                                                    output_white_level=args.output_white_level,
                                                    gamma=args.gamma)
        else:
            logger.error(f"ToneMapper {args.tone_mapper} does not exists.")
            raise ValueError

        tone_mapper.tone_map(image)
    # ##################################################################################################################

    # Demosaicer
    # ##################################################################################################################
    if args.demosaicer is not None:
        blue_loc = (int(args.blue_loc[0]), int(args.blue_loc[1]))
        if args.demosaicer == 'bayer_splitter':
            demosaicer = BayerSplitter(blue_loc=blue_loc)
        elif args.demosaicer == 'copy':
            demosaicer = DemosaicerCopy(blue_loc=blue_loc)
        elif args.demosaicer == 'linear':
            demosaicer = DemosaicerLinear(blue_loc=blue_loc)
        else:
            logger.error(f"Demosaicer {args.demosaicer} does not exist.")
            raise ValueError

        demosaicer.demosaice(image)
    # ##################################################################################################################

    # WhiteBalancer
    # ##################################################################################################################
    if args.white_balancer is not None:
        if args.white_balancer == 'camera':
            white_balancer = WhiteBalancerCamera()
        elif args.white_balancer == 'white_patch':
            white_balancer = WhiteBalancerWhitePatch(args.percentile)
        elif args.white_balancer == 'gray_world':
            white_balancer = WhiteBalancerGrayWorld()
        else:
            logger.error(f"Rotator {args.white_balancer} does not exists.")
            raise ValueError

        white_balancer.white_balance(image)
    # ##################################################################################################################

    # Rotator
    # ##################################################################################################################
    if args.rotator is not None:
        if args.rotator == 'numpy90':
            rotator = RotatorNumPy90()
        else:
            logger.error(f"Rotator {args.rotator} does not exists.")
            raise ValueError

        rotator.rotate_k(image, k=args.k)
    # ##################################################################################################################

    # Exporter
    # ##################################################################################################################
    if args.exporter == 'open_cv':
        exporter = ExporterOpenCV()
    else:
        logger.error(f"Exporter {args.exporter} does not exists.")
        raise ValueError

    exporter.export(image, path_to_export_image=args.path_to_export_image)
    # ##################################################################################################################


if __name__ == '__main__':
    main()
