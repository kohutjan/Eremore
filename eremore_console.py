import logging
import argparse
import sys

from core.loader import LoaderRawPy
from edit.editor import Editor
from edit.demosaicer import Demosaicer
from edit.tone_mapper import ToneMapper
from edit.white_balancer import WhiteBalancer
from edit.rotator import Rotator
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
    group_demosaicer.add_argument('--blue-loc', default='11', choices=['00', '01', '10', '11'])

    group_white_balancer = parser.add_argument_group('WhiteBalancer')
    group_white_balancer.add_argument('--white-balancer', choices=['camera', 'white_patch', 'gray_world'])
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
    if args.loader == 'rawpy':
        loader = LoaderRawPy()
    else:
        logger.error(f"Loader {args.loader} does not exist.")
        raise ValueError

    image = loader.load(path_to_raw_image=args.path_to_raw_image)
    editor = Editor(name='editor', input_image=image)
    # ##################################################################################################################

    # ToneMapper
    # ##################################################################################################################
    if args.tone_mapper is not None:
        tone_mapper = ToneMapper(engine=args.tone_mapper)
        tone_mapper.engines[tone_mapper.engine].input_black_level_correction = args.input_black_level_correction
        tone_mapper.engines[tone_mapper.engine].input_black_level = args.input_black_level
        tone_mapper.engines[tone_mapper.engine].input_white_level = args.input_white_level
        tone_mapper.engines[tone_mapper.engine].output_black_level = args.output_black_level
        tone_mapper.engines[tone_mapper.engine].output_white_level = args.output_white_level
        if tone_mapper.engine == 'gamma_correction':
            tone_mapper.engines['gamma_correction'].gamma = args.gamma

        editor.add_engine(tone_mapper)
        editor.register_engine_for_update(tone_mapper.name)
    # ##################################################################################################################

    # Demosaicer
    # ##################################################################################################################
    if args.demosaicer is not None:
        blue_loc = (int(args.blue_loc[0]), int(args.blue_loc[1]))
        demosaicer = Demosaicer(engine=args.demosaicer)
        demosaicer.engines[demosaicer.engine].blue_loc = blue_loc

        editor.add_engine(demosaicer)
        editor.register_engine_for_update(demosaicer.name)
    # ##################################################################################################################

    # WhiteBalancer
    # ##################################################################################################################
    if args.white_balancer is not None:
        white_balancer = WhiteBalancer(engine=args.white_balancer)
        if white_balancer.engine == 'white_patch':
            white_balancer.engines['white_patch'].percentile = args.percentile

        editor.add_engine(white_balancer)
        editor.register_engine_for_update(white_balancer.name)
    # ##################################################################################################################

    # Rotator
    # ##################################################################################################################
    if args.rotator is not None:
        rotator = Rotator(engine=args.rotator)
        if rotator.engine == '90':
            rotator.engines['90'].k = args.k

        editor.add_engine(rotator)
        editor.register_engine_for_update(rotator.name)
    # ##################################################################################################################

    editor.process()

    # Exporter
    # ##################################################################################################################
    if args.exporter == 'open_cv':
        exporter = ExporterOpenCV()
    else:
        logger.error(f"Exporter {args.exporter} does not exists.")
        raise ValueError

    exporter.export(editor.output_image, path_to_export_image=args.path_to_export_image)
    # ##################################################################################################################


if __name__ == '__main__':
    main()
