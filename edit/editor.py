import logging

from collections import OrderedDict
from copy import deepcopy

from core.image import Image

from helper.run_and_measure_time import run_and_measure_time


class Editor:
    def __init__(self, name: str, input_image: Image):
        self.logger = logging.getLogger(f"eremore.{__name__}.{name}")
        self.name = name
        self.input_image = input_image
        self.inputs = OrderedDict()
        self.engines = OrderedDict()
        self.engines_update_state = OrderedDict()
        self.output_image = None

    def add_engine(self, engine, engine_name=None):
        if engine_name is None:
            engine_name = engine.name
        self.engines[engine_name] = engine
        self.engines_update_state[engine_name] = False

    def register_engine_for_update(self, engine_name):
        self.engines_update_state[engine_name] = True

    def process(self):
        run_and_measure_time(self._process, {}, logger=self.logger)

    def _process(self):
        engines_to_update = self.get_engines_to_update()
        if engines_to_update[0] not in self.inputs:
            self.inputs[engines_to_update[0]] = self.input_image
        for i in range(len(engines_to_update)):
            output_image = deepcopy(self.inputs[engines_to_update[i]])
            self.engines[engines_to_update[i]].process(output_image)
            if i < len(engines_to_update) - 1:
                self.inputs[engines_to_update[i + 1]] = output_image
            else:
                self.output_image = output_image
            self.engines_update_state[engines_to_update[i]] = False

    def get_engines_to_update(self):
        engines_to_update = []
        update = False
        for engine_name, engine_update_state in self.engines_update_state.items():
            if engine_update_state or update:
                update = True
                engines_to_update.append(engine_name)
        return engines_to_update
