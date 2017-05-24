import functools
import operator

import numpy as np

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

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
        self.dtype = np.uint8  # HARDCODED
        self.buffers = [0] * 9  # HARDCODED
        self.frame_buf = np.zeros((3, ), dtype=np.uint8)
        self.size = experiment_class.size
        cells_total = functools.reduce(operator.mul, self.size)
        cells_total *= len(self.buffers) + 1
        self.cells_gpu = gpuarray.zeros((cells_total,), dtype=self.dtype)

    def set_viewport(self, size):
        self.width, self.height = w, h = size
        self.frame_buf = np.zeros((w * h * 3,), dtype=np.uint8)
        self.img_gpu = gpuarray.to_gpu(self.frame_buf)

    def step(self):
        self.frame_buf = np.random.randint(0, 255,
                                           self.frame_buf.shape,
                                           dtype=np.uint8)

    def render(self):
        return self.frame_buf
