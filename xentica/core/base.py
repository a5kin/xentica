"""
The module with the base class for all CA models.

All Xentica models should be inherited from :class:`CellularAutomaton`
base class. Inside the model, you should correctly define the
``Topology`` class and describe the CA logic in ``emit()``,
``absorb()`` and ``color()`` methods.

``Topology`` is the place where you define the dimensionality,
lattice, neighborhood and border effects for your CA. See
:mod:`xentica.core.topology` package for details.

The logic of the model will follow Buffered State Cellular Automaton
(BSCA) principle. In general, every cell mirrors its state in buffers
by the number of neighbors, each buffer intended for one of
neighbors. Then, at each step, the interaction between cells are
performed via buffers in 2-phase emit/absorb process. More detailed
description of BSCA principle is available in The Core section of `The
Concept`_ document.

``emit()`` describes the logic of the first phase of BSCA. At this
phase, you should fill cell's buffers with corresponding values,
depending on cell's main state and (optionally) on neighbors' main
states. The most easy logic is to just copy the main state to
buffers. It is esspecially useful when you're intending to emulate
classic CA (like Conway's Life) with BSCA. Write access to main state
is prohibited there.

``absorb()`` describes the logic of the second phase of BSCA. At this
phase, you should set the cell's main state, depending on neighbors'
buffered states. Write access to buffers is prohibited there.

``color()`` describes how to calculate cell's color from its raw
state. See detailed instructions on it in :mod:`xentica.core.color_effects`.

The logic of the functions from above will be translated into C code
at the moment of class creation. For the further instructions on how
to use cell's main and buffered states, see
:mod:`xentica.core.properties`, for the instructions on variables and
expressions with them, see :mod:`xentica.core.variables`.

A minimal example, the CA where each cell is taking the mean value of
its neighbors each step::

    from xentica import core
    from xentica.core import color_effects

    class MeanCA(core.CellularAutomaton):

        state = core.IntegerProperty(max_val=255)

        class Topology:
            dimensions = 2
            lattice = core.OrthogonalLattice()
            neighborhood = core.MooreNeighborhood()
            border = core.TorusBorder()

        def emit(self):
            for i in range(len(self.buffers)):
                self.buffers[i].state = self.main.state

        def absorb(self):
            s = core.IntegerVariable()
            for i in range(len(self.buffers)):
                s += self.neighbors[i].buffer.state
            self.main.state = s / len(self.buffers)

        @color_effects.MovingAverage
        def color(self):
            v = self.main.state
            return (v, v, v)

.. _The Concept: http://artipixoids.a5kin.net/concept/artipixoids_concept.pdf

"""
import functools
import itertools
import operator
import threading
import pickle
import collections
from copy import deepcopy

import numpy as np

from pycuda.autoinit import context
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

from xentica.bridge import MoireBridge
from xentica.seeds.random import LocalRandom
from xentica.core.properties import Property, ContainerProperty
from xentica.core.renderers import RendererPlain
from xentica.core.exceptions import XenticaException

__all__ = ['context', 'BSCA', 'CellularAutomaton', 'CachedNeighbor']


class CachedNeighbor:
    """Utility class, intended to hold main and buffered CA state."""

    def __init__(self):
        """Initialize empty main and buffered states."""
        self.main = ContainerProperty()
        self.buffer = ContainerProperty()


class BSCA(type):
    """
    Meta-class for CellularAutomaton.

    Generates parallel code given class definition
    and compiles it for future use.

    """
    @classmethod
    def __prepare__(self, name, bases):
        return collections.OrderedDict()

    def __new__(cls, name, bases, attrs):
        # prepare new class
        keys = []
        for key in attrs.keys():
            if key not in ('__module__', '__qualname__'):
                keys.append(key)
        attrs['__ordered__'] = keys
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
            raise XenticaException("No Topology class declared.")

        mandatory_fields = (
            'dimensions', 'lattice', 'neighborhood', 'border',
        )
        for f in mandatory_fields:
            if not hasattr(cls.topology, f):
                raise XenticaException("No %s declared in Topology class." % f)

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
        restricted_names = {"main", "buffer"}
        for obj_name, obj in itertools.chain.from_iterable(attrs_items):
            if isinstance(obj, Property) and obj_name not in restricted_names:
                cls._new_class.main[obj_name] = deepcopy(obj)
                vname = "_cell_%s" % obj_name
                cls._new_class.main[obj_name].var_name = vname
                for i in range(num_neighbors):
                    buffers = cls._new_class.buffers
                    buffers[i][obj_name] = deepcopy(obj)
                    vname = "_bcell_%s%d" % (obj_name, i)
                    buffers[i][obj_name].var_name = vname
                    neighbor = cls._new_class.neighbors[i]
                    neighbor.main[obj_name] = deepcopy(obj)
                    vname = "_dcell_%s%d" % (obj_name, i)
                    neighbor.main[obj_name].var_name = vname
                    neighbor.buffer[obj_name] = deepcopy(obj)
                    vname = "_dbcell_%s%d" % (obj_name, i)
                    neighbor.buffer[obj_name].var_name = vname
        # propagade BSCA to properties
        cls._new_class.main.set_bsca(cls._new_class, 0, -1)
        cls._new_class.main.var_name = "_cell"
        for i in range(num_neighbors):
            cls._new_class.buffers[i].set_bsca(cls._new_class, i + 1, -1)
            cls._new_class.buffers[i].var_name = "_bcell%i" % i
            cls._new_class.neighbors[i].main.set_bsca(cls._new_class, 0, i)
            cls._new_class.neighbors[i].main.var_name = "_dcell%d" % i
            cls._new_class.neighbors[i].buffer.set_bsca(cls._new_class,
                                                        i + 1, i)
            cls._new_class.neighbors[i].buffer.var_name = "_dbcell%d" % i

        cls._new_class.dtype = cls._new_class.main.dtype
        cls._new_class.ctype = cls._new_class.main.ctype

        cls._new_class._constants = {}

        # set default renderer as needed
        if not hasattr(cls._new_class, 'renderer'):
            cls._new_class.renderer = RendererPlain()

        # build CUDA source
        source = cls._new_class.build_emit()
        source += cls._new_class.build_absorb()
        source += cls._new_class.build_render()
        source = cls._new_class.build_defines() + source
        cls._new_class.cuda_source = source

        cls._new_class.index_to_coord = cls.index_to_coord
        cls._new_class.pack_state = cls.pack_state

        cls._new_class.size = (1 for i in range(cls.topology.dimensions))

        return cls._new_class

    def index_to_coord(cls, i):
        return cls.topology.lattice.index_to_coord(i, cls)

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
        cls._unpacks = set()
        cls._coords_declared = False
        for func in funcs:
            func(cls)
        for p in cls._deferred_writes:
            p.deferred_write()
        return cls._func_body

    def append_code(cls, code):
        cls._func_body += code

    def deferred_write(cls, prop):
        cls._deferred_writes.add(prop)

    def declare(cls, prop):
        cls._declarations.add(prop)

    def unpack(cls, prop):
        cls._unpacks.add(prop)

    def is_declared(cls, prop):
        return prop in cls._declarations

    def is_unpacked(cls, prop):
        return prop in cls._unpacks

    def declare_coords(cls):
        cls._coords_declared = True

    def define_constant(cls, constant):
        cls._constants[constant.name] = deepcopy(constant)

    def is_constant(cls, constant):
        return constant in cls._constants

    @property
    def coords_declared(cls):
        return cls._coords_declared

    def build_defines(cls):
        defines = ""
        for c in cls._constants.values():
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
        args = cls.renderer.args
        body = cls.renderer.render_code()
        return cls._elementwise_kernel("render", args, body)

    def pack_state(self, state):
        val = 0
        shift = 0
        for name, prop in self.main._properties.items():
            val += state[name] << shift
            shift += prop.bit_width
        return val


class CellularAutomaton(metaclass=BSCA):
    """
    Base class for all Xentica mods.

    """
    def __init__(self, experiment_class):
        # visuals
        self.frame_buf = np.zeros((3, ), dtype=np.uint8)
        # populate attributes from Experiment class
        for attr_name in dir(experiment_class):
            attr = getattr(experiment_class, attr_name)
            if (not callable(attr) and not attr_name.startswith("__")):
                if attr_name == 'seed':
                    continue
                setattr(self, attr_name, attr)
        # default simulation values
        self.speed = 1
        self.paused = False
        self.timestep = 0
        # CUDA kernel
        self.cells_num = functools.reduce(operator.mul, self.size)
        source = self.cuda_source
        for c in self._constants.values():
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
        self.renderer.setup_actions(self.bridge)
        # lock
        self.lock = threading.Lock()

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
            args = self.renderer.get_args_vals(self)
            args.append(np.int32(self.width * self.height))
            self.render_gpu(*args, block=block, grid=grid)
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
