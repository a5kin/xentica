import functools
import itertools
import operator
import threading
import pickle
from copy import deepcopy

import numpy as np

from pycuda.autoinit import context
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

from hecate.bridge import MoireBridge
from hecate.seeds.random import LocalRandom
from hecate.core.properties import Property, ContainerProperty
from hecate.core.variables import Constant

__all__ = ['context', ]


class HecateException(Exception):
    """ Basic Hecate framework exception """


class CachedNeighbor:

    def __init__(self):
        self.main = ContainerProperty()
        self.buffer = ContainerProperty()


class BSCA(type):
    """
    Meta-class for CellularAutomaton.

    Generates parallel code given class definition
    and compiles it for future use.

    """
    def __new__(cls, name, bases, attrs):
        # prepare new class
        cls._new_class = super().__new__(cls, name, bases, attrs)
        cls._parents = [b for b in bases if isinstance(b, BSCA)]
        if not cls._parents:
            return cls._new_class

        # prepare topology
        if hasattr(cls._new_class, 'Topology'):
            attrs['Topology'] = cls._new_class.Topology
        cls.topology = attrs.get('Topology', None)
        cls._new_class.topology = cls.topology

        if cls.topology is None:
            raise HecateException("No Topology class declared.")

        mandatory_fields = (
            'dimensions', 'lattice', 'neighborhood', 'border',
        )
        for f in mandatory_fields:
            if not hasattr(cls.topology, f):
                raise HecateException("No %s declared in Topology class." % f)

        cls.topology.lattice.dimensions = cls.topology.dimensions
        cls.topology.neighborhood.dimensions = cls.topology.dimensions
        cls.topology.border.dimensions = cls.topology.dimensions
        cls.topology.neighborhood.topology = cls.topology
        cls.topology.border.topology = cls.topology

        # scan and prepare properties
        cls._new_class.main = ContainerProperty()
        cls._new_class.buffers = []
        cls._new_class.neighbors = []
        num_neighbors = len(cls.topology.neighborhood)
        for i in range(num_neighbors):
            cls._new_class.buffers.append(ContainerProperty())
            cls._new_class.neighbors.append(CachedNeighbor())
        attrs_items = [base_class.__dict__.items() for base_class in bases]
        attrs_items.append(attrs.items())
        for obj_name, obj in itertools.chain.from_iterable(attrs_items):
            if isinstance(obj, Property):
                cls._new_class.main[obj_name] = deepcopy(obj)
                cls._new_class.main[obj_name].var_name = "_cell"
                for i in range(num_neighbors):
                    buffers = cls._new_class.buffers
                    buffers[i][obj_name] = deepcopy(obj)
                    buffers[i][obj_name].var_name = "_bcell%i" % i
                    neighbor = cls._new_class.neighbors[i]
                    neighbor.main[obj_name] = deepcopy(obj)
                    neighbor.main[obj_name].var_name = "_dcell%d" % i
                    neighbor.buffer[obj_name] = deepcopy(obj)
                    neighbor.buffer[obj_name].var_name = "_dbcell%d" % i
        # propagade BSCA to properties
        cls._new_class.main.set_bsca(cls._new_class, 0, -1)
        for i in range(num_neighbors):
            cls._new_class.buffers[i].set_bsca(cls._new_class, i + 1, -1)
            cls._new_class.neighbors[i].main.set_bsca(cls._new_class, 0, i)
            cls._new_class.neighbors[i].buffer.set_bsca(cls._new_class,
                                                        i + 1, i)

        cls._new_class.dtype = cls._new_class.main.dtype
        cls._new_class.ctype = cls._new_class.main.ctype

        cls._new_class._constants = set()
        # hardcoded constants
        for i in range(cls._new_class.topology.dimensions):
            cls._new_class._constants.add(Constant("_w%d" % i, "size[%d]" % i))
        cls._new_class._constants.add(Constant("FADE_IN", "fade_in"))
        cls._new_class._constants.add(Constant("FADE_OUT", "fade_out"))
        cls._new_class._constants.add(Constant("SMOOTH_FACTOR",
                                               "smooth_factor"))

        # build CUDA source
        source = cls._new_class.build_emit()
        source += cls._new_class.build_absorb()
        source += cls._new_class.build_render()
        source = cls._new_class.build_defines() + source
        cls._new_class.cuda_source = source

        cls._new_class.index_to_coord = cls.index_to_coord
        cls._new_class.pack_state = cls.pack_state
        cls._new_class.topology = cls.topology

        # hardcoded stuff
        cls._new_class.fade_in = 255
        cls._new_class.fade_out = 255
        cls._new_class.smooth_factor = 1
        return cls._new_class

    def _elementwise_kernel(self, name, args, body):
        arg_string = ", ".join(["%s %s" % (t, v) for t, v in args])
        kernel = """
            __global__ void %s(%s, int n) {

                unsigned tid = threadIdx.x;
                unsigned total_threads = gridDim.x * blockDim.x;
                unsigned cta_start = blockDim.x * blockIdx.x;

                for (unsigned i = cta_start + tid; i < n; i += total_threads) {
                    %s
                }

            }

        """ % (name, arg_string, body)
        return kernel

    def _translate_code(cls, *funcs):
        cls._func_body = ""
        cls._deferred_writes = set()
        cls._declarations = set()
        cls._coords_declared = False
        for func in funcs:
            func(cls)
        for p in cls._deferred_writes:
            cls._func_body += "%s = %s;\n" % (p._mem_cell, p.var_name)
        return cls._func_body

    def append_code(cls, code):
        cls._func_body += code

    def deferred_write(cls, prop):
        cls._deferred_writes.add(prop)

    def declare(cls, prop):
        cls._declarations.add(prop)

    def is_declared(cls, prop):
        return prop in cls._declarations

    def declare_coords(cls):
        cls._coords_declared = True

    @property
    def coords_declared(cls):
        return cls._coords_declared

    def build_defines(cls):
        defines = ""
        for c in cls._constants:
            defines += c.get_define_code()
        return defines

    def build_emit(cls):
        args = [(cls.ctype, "*fld"), ]
        body = cls._translate_code(cls.emit)
        return cls._elementwise_kernel("emit", args, body)

    def build_absorb(cls):
        args = [(cls.ctype, "*fld"), ("int3", "*col")]
        body = cls._translate_code(cls.absorb, cls.color)
        return cls._elementwise_kernel("absorb", args, body)

    def build_render(cls):
        # hardcoded for now
        args = [
            ("int3", "*col"),
            ("int", "*img"),
            ("int", "zoom"),
            ("int", "dx"),
            ("int", "dy"),
            ("int", "width"),
        ]
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
        source = self.cuda_source
        for c in self._constants:
            source = c.replace_value(source)
        # print(source)
        cuda_module = SourceModule(source)
        # GPU arrays
        self.emit_gpu = cuda_module.get_function("emit")
        self.absorb_gpu = cuda_module.get_function("absorb")
        self.render_gpu = cuda_module.get_function("render")
        init_colors = np.zeros((self.cells_num * 3, ), dtype=np.int32)
        self.colors_gpu = gpuarray.to_gpu(init_colors)
        cells_total = self.cells_num * (len(self.buffers) + 1) + 1
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
