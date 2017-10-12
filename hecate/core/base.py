import functools
import operator
import threading
import pickle

import numpy as np

from pycuda.autoinit import context
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

from hecate.bridge import MoireBridge
from hecate.seeds.random import LocalRandom

__all__ = ['context', ]


class HecateException(Exception):
    """ Basic Hecate framework exception """


class BSCA(type):
    """
    Meta-class for CellularAutomaton.

    Generates parallel code given class definition
    and compiles it for future use.

    """
    def __new__(cls, name, bases, attrs):
        cls._new_class = super().__new__(cls, name, bases, attrs)
        cls._parents = [b for b in bases if isinstance(b, BSCA)]
        if not cls._parents:
            return cls._new_class

        cls._topology = attrs.get('Topology', None)
        cls._new_class._topology = cls._topology
        if cls._topology is None:
            raise HecateException("No Topology class declared.")

        mandatory_fields = (
            'dimensions', 'lattice', 'neighborhood', 'border',
        )
        for f in mandatory_fields:
            if not hasattr(cls._topology, f):
                raise HecateException("No %s declared in Topology class." % f)

        cls._topology.lattice.dimensions = cls._topology.dimensions
        cls._topology.neighborhood.dimensions = cls._topology.dimensions
        cls._topology.border.dimensions = cls._topology.dimensions
        cls._topology.neighborhood.topology = cls._topology
        cls._topology.border.topology = cls._topology

        # build CUDA source
        cls._new_class.cuda_source = cls._new_class._build_defines()
        cls._new_class.cuda_source += cls._new_class._build_emit()
        cls._new_class.cuda_source += cls._new_class._build_absorb()
        cls._new_class.cuda_source += cls._new_class._build_render()
        cls._new_class.index_to_coord = cls.index_to_coord
        cls._new_class.pack_state = cls.pack_state
        cls._new_class._topology = cls._topology

        # hardcoded stuff
        cls._new_class.dtype = np.uint8
        cls._new_class.buffers = [0] * 9
        cls._new_class.fade_in = 255
        cls._new_class.fade_out = 255
        cls._new_class.smooth_factor = 1
        return cls._new_class

    def _elementwise_kernel(self, name, args, body):
        kernel = """
            __global__ void %s(%s, int n) {

                unsigned tid = threadIdx.x;
                unsigned total_threads = gridDim.x * blockDim.x;
                unsigned cta_start = blockDim.x * blockIdx.x;

                for (unsigned i = cta_start + tid; i < n; i += total_threads) {
                    %s
                }

            }

        """ % (name, args, body)
        return kernel

    def _translate_code(cls, func):
        # hardcoded for now
        if func.__name__ == 'emit':
            body = ""
            for i in range(len(cls._topology.neighborhood)):
                body += "fld[i + n * %d] = fld[i];\n" % (i + 1, )
            return body
        if func.__name__ == 'absorb':
            num_neighbors = len(cls._topology.neighborhood)
            neighbors = ["_dcell%d" % i for i in range(num_neighbors)]
            return """
                unsigned char s = {summed_neighbors};
                unsigned char state;
                state = ((8 >> s) & 1) | ((12 >> s) & 1) & fld[i + n];
                fld[i] = state;
            """.format(summed_neighbors=" + ".join(neighbors))
        if func.__name__ == 'color':
            return """
                int new_r = state * 255 * SMOOTH_FACTOR;
                int new_g = state * 255 * SMOOTH_FACTOR;
                int new_b = state * 255 * SMOOTH_FACTOR;
                int3 old_col = col[i];
                new_r = max(min(new_r, old_col.x + FADE_IN),
                            old_col.x - FADE_OUT);
                new_g = max(min(new_g, old_col.y + FADE_IN),
                            old_col.y - FADE_OUT);
                new_b = max(min(new_b, old_col.z + FADE_IN),
                            old_col.z - FADE_OUT);
                col[i] = make_int3(new_r, new_g, new_b);
            """

    def _build_defines(cls):
        defines = ""
        for i in range(cls._topology.dimensions):
            defines += "#define _w%d {w%d}\n" % (i, i)
        # hardcoded for now
        defines += """
            #define FADE_IN {fadein}
            #define FADE_OUT {fadeout}
            #define SMOOTH_FACTOR {smooth}

        """
        return defines

    def _build_emit(cls):
        args = "unsigned char *fld"
        body = cls._translate_code(cls.emit)
        return cls._elementwise_kernel("emit", args, body)

    def _build_absorb(cls):
        args = "unsigned char *fld, int3 *col"
        body = cls._topology.lattice.index_to_coord_code("i", "_x")
        coord_vars = ["_nx%d" % i for i in range(cls._topology.dimensions)]
        neighborhood = cls._topology.neighborhood
        body += "int %s;\n" % ", ".join(coord_vars)
        for i in range(len(cls._topology.neighborhood)):
            body += neighborhood.neighbor_coords(i, "_x", "_nx")
            state_code = neighborhood.neighbor_state(i, i, "_nx",
                                                     "_dcell%d" % i)
            is_cell_off_board = cls._topology.lattice.is_off_board_code("_nx")
            body += "unsigned char _dcell%d;" % i
            if hasattr(cls._topology.border, "wrap_coords"):
                body += """
                    if ({is_cell_off_board}) {{
                        {wrap_coords}
                    }}
                """.format(
                    is_cell_off_board=is_cell_off_board,
                    wrap_coords=cls._topology.border.wrap_coords("_nx"),
                )
                body += state_code
            else:
                body += """
                    if ({is_cell_off_board}) {{
                        {off_board_cell}
                    }} else {{
                        {get_neighbor_state}
                    }}
                """.format(
                    is_cell_off_board=is_cell_off_board,
                    off_board_cell=cls._topology.border.off_board_state(
                        "_nx", "_dcell%d" % i
                    ),
                    get_neighbor_state=state_code,
                )
        body += cls._translate_code(cls.absorb)
        body += cls._translate_code(cls.color)
        return cls._elementwise_kernel("absorb", args, body)

    def _build_render(cls):
        args = "int3 *col, int *img, int zoom, int dx, int dy, int width"
        # hardcoded for now
        body = """
            int x = (int) (((float) (i % width)) / (float) zoom) + dx;
            int y = (int) (((float) (i / width)) / (float) zoom) + dy;
            if (x < 0) x = _w0 - (-x % _w0);
            if (x >= _w0) x = x % _w0;
            if (y < 0) y = _w1 - (-y % _w1);
            if (y >= _w1) y = y % _w1;
            int ii = x + y * _w0;

            int3 c = col[ii];
            img[i * 3] = c.x / SMOOTH_FACTOR;
            img[i * 3 + 1] = c.y / SMOOTH_FACTOR;
            img[i * 3 + 2] = c.z / SMOOTH_FACTOR;
        """
        return cls._elementwise_kernel("render", args, body)

    def index_to_coord(self, i):
        return (i % self.size[0], i // self.size[0])

    def pack_state(self, state):
        return state['state']


class CellularAutomaton(metaclass=BSCA):
    """
    Base class for all HECATE mods.

    """
    def __init__(self, experiment_class):
        # visuals
        self.frame_buf = np.zeros((3, ), dtype=np.uint8)
        self.size = experiment_class.size
        self.zoom = experiment_class.zoom
        self.pos = experiment_class.pos
        self.speed = 1
        self.paused = False
        self.timestep = 0
        # CUDA kernel
        self.cells_num = functools.reduce(operator.mul, self.size)
        source = self.cuda_source.replace("{n}", str())
        for i in range(self._topology.dimensions):
            source = source.replace("{w%d}" % i, str(self.size[i]))
        source = source.replace("{fadein}", str(self.fade_in))
        source = source.replace("{fadeout}", str(self.fade_out))
        source = source.replace("{smooth}", str(self.smooth_factor))
        cuda_module = SourceModule(source)
        # GPU arrays
        self.emit_gpu = cuda_module.get_function("emit")
        self.absorb_gpu = cuda_module.get_function("absorb")
        self.render_gpu = cuda_module.get_function("render")
        init_colors = np.zeros((self.cells_num * 3, ), dtype=np.int32)
        self.colors_gpu = gpuarray.to_gpu(init_colors)
        cells_total = self.cells_num * len(self.buffers) + 1
        self.random = LocalRandom(experiment_class.word)
        experiment_class.seed.random = self.random
        init_cells = np.zeros((cells_total, ), dtype=self.dtype)
        experiment_class.seed.generate(init_cells, self.cells_num,
                                       self.size, self.index_to_coord,
                                       self.pack_state)
        self.cells_gpu = gpuarray.to_gpu(init_cells)
        # bridge
        self.bridge = MoireBridge
        # lock
        self.lock = threading.Lock()

    def move(self, *args):
        for i in range(len(args)):
            delta = args[i]
            self.pos[i] = (self.pos[i] + delta) % self.size[i]

    def apply_zoom(self, dval):
        self.zoom = max(1, (self.zoom + dval))

    def apply_speed(self, dval):
        self.speed = max(1, (self.speed + dval))

    def toggle_pause(self):
        self.paused = not self.paused

    def set_viewport(self, size):
        self.width, self.height = w, h = size
        self.img_gpu = gpuarray.zeros((w * h * 3), dtype=np.int32)

    def step(self):
        if self.paused:
            return
        block, grid = self.cells_gpu._block, self.cells_gpu._grid
        with self.lock:
            self.emit_gpu(self.cells_gpu, np.int32(self.cells_num),
                          block=block, grid=grid)
            self.absorb_gpu(self.cells_gpu, self.colors_gpu,
                            np.int32(self.cells_num),
                            block=block, grid=grid)
            self.timestep += 1

    def render(self):
        block, grid = self.img_gpu._block, self.img_gpu._grid
        with self.lock:
            self.render_gpu(self.colors_gpu, self.img_gpu,
                            np.int32(self.zoom),
                            np.int32(self.pos[0]), np.int32(self.pos[1]),
                            np.int32(self.width),
                            np.int32(self.width * self.height),
                            block=block, grid=grid)
            return self.img_gpu.get().astype(np.uint8)

    def save(self, filename):
        with open(filename, "wb") as f:
            ca_state = {
                "cells": self.cells_gpu.get(),
                "colors": self.colors_gpu.get(),
                "random": self.random,
            }
            pickle.dump(ca_state, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            ca_state = pickle.load(f)
            self.cells_gpu = gpuarray.to_gpu(ca_state['cells'])
            self.colors_gpu = gpuarray.to_gpu(ca_state['colors'])
            self.random.load(ca_state['random'])
