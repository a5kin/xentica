import numpy as np

class BSCA(type):
    """
    Meta-class for CellularAutomaton.

    Generates parallel code given class definition
    and compiles it for future use.

    """


class CellularAutomaton(metaclass=BSCA):
    """
    Base class for all HECATE mods.

    """
    def __init__(self, experiment_class):
        self.viewport_buf = np.zeros((3, ), dtype=np.uint8)

    def set_viewport(self, size):
        w, h = size
        self.viewport_buf = np.zeros((w * h * 3,), dtype=np.uint8)

    def step(self):
        self.viewport_buf = np.random.randint(0, 255,
                                              self.viewport_buf.shape,
                                              dtype=np.uint8)

    def render(self):
        return self.viewport_buf
