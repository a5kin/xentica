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
    def __init__(cls, name, bases, namespace, **kwds1):
        # hardcoded stuff
        cls.dtype = np.uint8
        cls.buffers = [0] * 9
        cls.cuda_source = """
            #define w {w}
            #define h {h}
            #define n {n}

            __global__ void emit(unsigned char *fld) {

                unsigned tid = threadIdx.x;
                unsigned total_threads = gridDim.x * blockDim.x;
                unsigned cta_start = blockDim.x * blockIdx.x;
                unsigned i;

                for (i = cta_start + tid; i < n; i += total_threads) {

                    fld[i + n] = fld[i];

                }

            }

            __global__ void absorb(unsigned char *fld) {

                unsigned tid = threadIdx.x;
                unsigned total_threads = gridDim.x * blockDim.x;
                unsigned cta_start = blockDim.x * blockIdx.x;
                unsigned i;

                for (i = cta_start + tid; i < n; i += total_threads) {

                    int x = i % w;
                    int y = i / w;
                    int xm1 = x - 1; if (xm1 < 0) xm1 = w + xm1;
                    int xp1 = x + 1; if (xp1 >= w) xp1 = xp1 - w;
                    int ym1 = y - 1; if (ym1 < 0) ym1 = h + ym1;
                    int yp1 = y + 1; if (yp1 >= h) yp1 = yp1 - h;
                    unsigned char s = fld[xm1 + ym1 * w + n] +
                                      fld[x + ym1 * w + n] +
                                      fld[xp1 + ym1 * w + n] +
                                      fld[xm1 + y * w + n] +
                                      fld[xp1 + y * w + n] +
                                      fld[xm1 + yp1 * w + n] +
                                      fld[x + yp1 * w + n] +
                                      fld[xp1 + yp1 * w + n];
                    fld[i] = ((8 >> s) & 1) | ((12 >> s) & 1) & fld[i];

                }

            }
        """


class CellularAutomaton(metaclass=BSCA):
    """
    Base class for all HECATE mods.

    """
    def __init__(self, experiment_class):
        self.frame_buf = np.zeros((3, ), dtype=np.uint8)
        self.size = experiment_class.size
        cells_total = functools.reduce(operator.mul, self.size)
        source = self.cuda_source.replace("{n}", str(cells_total))
        source = source.replace("{w}", str(self.size[0]))
        source = source.replace("{h}", str(self.size[1]))
        cuda_module = SourceModule(source)
        self.emit_gpu = cuda_module.get_function("emit")
        self.absorb_gpu = cuda_module.get_function("absorb")
        cells_total *= len(self.buffers) + 1
        init_cells = np.random.randint(2, size=cells_total, dtype=self.dtype)
        self.cells_gpu = gpuarray.to_gpu(init_cells)

    def set_viewport(self, size):
        self.width, self.height = w, h = size
        frame_buf = np.zeros((w * h * 3,), dtype=np.uint8)
        self.img_gpu = gpuarray.to_gpu(frame_buf)

    def step(self):
        block, grid = self.cells_gpu._block, self.cells_gpu._grid
        self.emit_gpu(self.cells_gpu, block=block, grid=grid)
        self.absorb_gpu(self.cells_gpu, self.img_gpu, block=block, grid=grid)

    def render(self):
        return self.cells_gpu.get() * 255
