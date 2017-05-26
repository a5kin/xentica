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
        cls.fade_in = 255
        cls.fade_out = 10
        cls.smooth_factor = 1
        cls.cuda_source = """
            #define w {w}
            #define h {h}
            #define n {n}
            #define FADE_IN {fadein}
            #define FADE_OUT {fadeout}
            #define SMOOTH_FACTOR {smooth}

            __global__ void emit(uchar1 *fld) {

                unsigned tid = threadIdx.x;
                unsigned total_threads = gridDim.x * blockDim.x;
                unsigned cta_start = blockDim.x * blockIdx.x;
                unsigned i;

                for (i = cta_start + tid; i < n; i += total_threads) {

                    fld[i + n] = fld[i];

                }

            }

            __global__ void absorb(unsigned char *fld, int *col) {

                unsigned tid = threadIdx.x;
                unsigned total_threads = gridDim.x * blockDim.x;
                unsigned cta_start = blockDim.x * blockIdx.x;
                unsigned i;

                for (i = cta_start + tid; i < n; i += total_threads) {

                    unsigned x = i % w;
                    unsigned y = i / w;
                    unsigned xm1 = x - 1; if (xm1 < 0) xm1 = w + xm1;
                    unsigned xp1 = x + 1; if (xp1 >= w) xp1 = xp1 - w;
                    unsigned ym1 = y - 1; if (ym1 < 0) ym1 = h + ym1;
                    unsigned yp1 = y + 1; if (yp1 >= h) yp1 = yp1 - h;
                    unsigned char s = fld[xm1 + ym1 * w + n] +
                                      fld[x + ym1 * w + n] +
                                      fld[xp1 + ym1 * w + n] +
                                      fld[xm1 + y * w + n] +
                                      fld[xp1 + y * w + n] +
                                      fld[xm1 + yp1 * w + n] +
                                      fld[x + yp1 * w + n] +
                                      fld[xp1 + yp1 * w + n];
                    unsigned char state;
                    state = ((8 >> s) & 1) | ((12 >> s) & 1) & fld[i];
                    fld[i] = state;

                    int new_r = state * 255 * SMOOTH_FACTOR;
                    int new_g = state * 255 * SMOOTH_FACTOR;
                    int new_b = state * 255 * SMOOTH_FACTOR;
                    int old_r = col[i * 3];
                    int old_g = col[i * 3 + 1];
                    int old_b = col[i * 3 + 2];
                    col[i * 3] = max(min(new_r, old_r + FADE_IN), old_r - FADE_OUT);
                    col[i * 3 + 1] = max(min(new_g, old_g + FADE_IN), old_g - FADE_OUT);
                    col[i * 3 + 2] = max(min(new_b, old_b + FADE_IN), old_b - FADE_OUT);

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
        source = source.replace("{fadein}", str(self.fade_in))
        source = source.replace("{fadeout}", str(self.fade_out))
        source = source.replace("{smooth}", str(self.smooth_factor))
        cuda_module = SourceModule(source)
        self.emit_gpu = cuda_module.get_function("emit")
        self.absorb_gpu = cuda_module.get_function("absorb")
        init_colors = np.zeros((cells_total * 3, ), dtype=np.uint32)
        self.colors_gpu = gpuarray.to_gpu(init_colors)
        cells_total *= len(self.buffers) + 1
        init_cells = np.random.randint(2, size=cells_total, dtype=self.dtype)
        self.cells_gpu = gpuarray.to_gpu(init_cells)

    def set_viewport(self, size):
        self.width, self.height = w, h = size
        # frame_buf = np.zeros((w * h * 3,), dtype=np.uint8)
        # self.img_gpu = gpuarray.to_gpu(frame_buf)

    def step(self):
        block, grid = self.cells_gpu._block, self.cells_gpu._grid
        self.emit_gpu(self.cells_gpu, block=block, grid=grid)
        self.absorb_gpu(self.cells_gpu, self.colors_gpu,
                        block=block, grid=grid)

    def render(self):
        return (self.colors_gpu.get() // self.smooth_factor).astype(np.uint8)
