import logging
import argparse
import sys

from functools import partial

from core.loader import Loader
from edit.editor import Editor
from edit.demosaicer import Demosaicer
from edit.tone_mapper import ToneMapper
from edit.white_balancer import WhiteBalancer
from edit.rotator import Rotator
from core.exporter import Exporter

logger = logging.getLogger(f"eremore.{__name__}")


def parseargs():
    print(' '.join(sys.argv))
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-magnitude', default=2**14, type=int)
    parser.add_argument('--input-black-level', default=0, type=int)
    parser.add_argument('--input-white-level', default=2**12-1, type=int)
    parser.add_argument('--output-black-level', default=0, type=int)
    parser.add_argument('--output-white-level', default=255, type=int)

    group_loader = parser.add_argument_group('Loader')
    group_loader.add_argument('--loader', default='raw_py', choices=['raw_py'])
    group_loader.add_argument('--path-to-raw-image', required=True, type=str, help="Path to the RAW image.")

    group_tone_mapper = parser.add_argument_group('ToneMapper')
    group_tone_mapper.add_argument('--tone-mapper', choices=['linear', 'gamma_correction'])
    group_tone_mapper.add_argument('--input-black-level-correction', default=0, type=int)

    group_tone_mapper_gamma_correction = parser.add_argument_group('ToneMapperGammaCorrection')
    group_tone_mapper_gamma_correction.add_argument('--gamma', default=1, type=float)

    group_demosaicer = parser.add_argument_group('Demosaicer')
    group_demosaicer.add_argument('--demosaicer', choices=['bayer_splitter', 'copy', 'linear'])
    group_demosaicer.add_argument('--blue-loc', default='11', choices=['00', '01', '10', '11'])

    group_white_balancer = parser.add_argument_group('WhiteBalancer')
    group_white_balancer.add_argument('--white-balancer', choices=['rgb', 'camera', 'white_patch', 'gray_world'])
    group_white_balancer_rgb = parser.add_argument_group('WhiteBalancerRGB')
    group_white_balancer_rgb.add_argument('--r-scale', default=1, type=int)
    group_white_balancer_rgb.add_argument('--g-scale', default=1, type=int)
    group_white_balancer_rgb.add_argument('--b-scale', default=1, type=int)
    group_white_balancer_white_patch = parser.add_argument_group('WhiteBalancerWhitePatch')
    group_white_balancer_white_patch.add_argument('--percentile', default=97, type=float)

    group_rotator = parser.add_argument_group('Rotator')
    group_rotator.add_argument('--rotator', choices=['90'])
    group_rotator_90 = parser.add_argument_group('Rotator90')
    group_rotator_90.add_argument('--k', type=int)

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
    loader = Loader(engine=args.loader)
    loader.engines[loader.engine].set(path_to_raw_image=args.path_to_raw_image)

    image = loader.process()
    # ##################################################################################################################

    editor = Editor(name='editor', input_image=image)

    # ToneMapper
    # ##################################################################################################################
    if args.tone_mapper is not None:
        tone_mapper = ToneMapper(engine=args.tone_mapper)
        tone_mapper_set = partial(tone_mapper.engines[tone_mapper.engine].set,
                                  input_magnitude=args.input_magnitude,
                                  input_black_level_correction=args.input_black_level_correction,
                                  input_black_level=args.input_black_level,
                                  input_white_level=args.input_white_level,
                                  output_black_level=args.input_black_level,
                                  output_white_level=args.input_white_level)
        if tone_mapper.engine == 'gamma_correction':
            tone_mapper_set(gamma=args.gamma)
        else:
            tone_mapper_set()

        editor.add_engine(tone_mapper)
        editor.register_engine_for_update(tone_mapper.name)
    # ##################################################################################################################

    # Demosaicer
    # ##################################################################################################################
    if args.demosaicer is not None:
        blue_loc = (int(args.blue_loc[0]), int(args.blue_loc[1]))
        demosaicer = Demosaicer(engine=args.demosaicer)
        demosaicer.engines[demosaicer.engine].set(blue_loc=blue_loc)

        editor.add_engine(demosaicer)
        editor.register_engine_for_update(demosaicer.name)
    # ##################################################################################################################

    # WhiteBalancer
    # ##################################################################################################################
    if args.white_balancer is not None:
        white_balancer = WhiteBalancer(engine=args.white_balancer)
        white_balancer_set = partial(white_balancer.engines[white_balancer.engine].set,
                                     input_magnitude=args.input_white_level + 1,
                                     input_black_level=args.input_black_level,
                                     input_white_level=args.input_white_level)
        if white_balancer.engine == 'white_patch':
            white_balancer_set(percentile=args.percentile)
        else:
            white_balancer_set()

        editor.add_engine(white_balancer)
        editor.register_engine_for_update(white_balancer.name)
    # ##################################################################################################################

    # Output Liner ToneMapper
    # ##################################################################################################################
    output_linear_tone_mapper = ToneMapper(name='output_linear_tone_mapper', engine='linear')
    output_linear_tone_mapper.engines[output_linear_tone_mapper.engine].set(name='output_linear',
                                                                            input_magnitude=args.input_magnitude,
                                                                            input_black_level_correction=0,
                                                                            input_black_level=args.input_black_level,
                                                                            input_white_level=args.input_white_level,
                                                                            output_black_level=args.output_black_level,
                                                                            output_white_level=args.output_white_level)

    editor.add_engine(output_linear_tone_mapper)
    editor.register_engine_for_update(output_linear_tone_mapper.name)
    # ##################################################################################################################

    # Rotator
    # ##################################################################################################################
    if args.rotator is not None:
        rotator = Rotator(engine=args.rotator)
        if rotator.engine == '90':
            rotator.engines['90'].set(k=args.k)

        editor.add_engine(rotator)
        editor.register_engine_for_update(rotator.name)
    # ##################################################################################################################


    editor.process()

    # Exporter
    # ##################################################################################################################
    exporter = Exporter(engine=args.exporter)
    exporter.engines[exporter.engine].set(path_to_export_image=args.path_to_export_image)

    exporter.process(editor.output_image)
    # ##################################################################################################################


if __name__ == '__main__':
    main()
