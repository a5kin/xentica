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
    Meta-class for :class:`CellularAutomaton`.

    Performs all necessary stuff to generate GPU kernels given class
    definition.

    It is also preparing ``main``, ``buffers`` and ``neighbors`` class
    variables being used in ``emit()``, ``absorb()`` and ``color()``
    methods.

    """

    mandatory_fields = (
        'dimensions', 'lattice', 'neighborhood', 'border',
    )

    @classmethod
    def __prepare__(mcs, _name, _bases):
        """Preserve the order of class variables."""
        return collections.OrderedDict()

    def __new__(mcs, name, bases, attrs):
        """Build new :class:`CellularAutomaton` class."""
        # prepare class itself
        keys = []
        for key in attrs.keys():
            if key not in ('__module__', '__qualname__'):
                keys.append(key)
        attrs['__ordered__'] = keys
        mcs._new_class = super().__new__(mcs, name, bases, attrs)
        mcs._parents = [b for b in bases if isinstance(b, BSCA)]
        if not mcs._parents:
            return mcs._new_class

        mcs._prepare_topology(mcs, attrs)

        mcs._new_class.main = None
        mcs._new_class.buffers = None
        mcs._new_class.neighbors = None
        mcs._prepare_properties(mcs, bases, attrs)

        mcs._new_class.dtype = mcs._new_class.main.dtype
        mcs._new_class.ctype = mcs._new_class.main.ctype

        mcs._new_class.constants = {}

        # set default renderer as needed
        if not hasattr(mcs._new_class, 'renderer'):
            mcs._new_class.renderer = RendererPlain()

        # build CUDA source
        source = mcs._new_class.build_emit()
        source += mcs._new_class.build_absorb()
        source += mcs._new_class.build_render()
        source = mcs._new_class.build_defines() + source
        mcs._new_class.cuda_source = source

        mcs._new_class.index_to_coord = mcs.index_to_coord
        mcs._new_class.pack_state = mcs.pack_state

        mcs._new_class.size = tuple(1 for i in range(mcs.topology.dimensions))

        return mcs._new_class

    def _prepare_topology(cls, attrs):
        """Prepare topology for future use."""
        if hasattr(cls._new_class, 'Topology'):
            attrs['Topology'] = cls._new_class.Topology
        cls.topology = attrs.get('Topology', None)
        cls._new_class.topology = cls.topology

        if cls.topology is None:
            raise XenticaException("No Topology class declared.")

        for field in cls.mandatory_fields:
            if not hasattr(cls.topology, field):
                msg = "No %s declared in Topology class." % field
                raise XenticaException(msg)

        cls.topology.lattice.dimensions = cls.topology.dimensions
        cls.topology.neighborhood.dimensions = cls.topology.dimensions
        cls.topology.border.dimensions = cls.topology.dimensions
        cls.topology.neighborhood.topology = cls.topology
        cls.topology.border.topology = cls.topology

    def _prepare_properties(cls, bases, attrs):
        """Prepare main/buffers properties."""
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

    def index_to_coord(cls, i):
        """
        Wrap ``lattice.index_to_coord`` method.

        :param i: Cell's index.

        """
        return cls.topology.lattice.index_to_coord(i, cls)

    @staticmethod
    def _elementwise_kernel(name, args, body):
        """
        Build elementwise kernel using template.

        :param name:
            Kernel's name.
        :param args:
            List of kernel's arguments, consisting of
            ``("type", "var_name")`` tuples.
        :param body:
            C code for the kernel's "body", processing a single element.

        """
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
        """
        Translate Python method to C code by execution.

        :param funcs:
            List of functions to translate in a single context.

        :returns:
            String with generated C code for elementwise kernel.

        """
        cls._func_body = ""
        cls._deferred_writes = set()
        cls._declarations = set()
        cls._unpacks = set()
        cls._coords_declared = False
        for func in funcs:
            func(cls)
        for prop in cls._deferred_writes:
            prop.deferred_write()
        return cls._func_body

    def append_code(cls, code):
        """Append ``code`` to kernel's C code."""
        cls._func_body += code

    def deferred_write(cls, prop):
        """
        Declare a property for deferred write.

        The property will be written into a memory at the end of C
        code generation.

        :param prop:
            :class:`Property <xentica.core.properties.Property>`
            subclass instance.

        """
        cls._deferred_writes.add(prop)

    def declare(cls, prop):
        """
        Mark property declared.

        :param prop:
            :class:`Property <xentica.core.properties.Property>`
            subclass instance.

        """
        cls._declarations.add(prop)

    def unpack(cls, prop):
        """
        Mark ``prop`` property unpacked.

        :param prop:
            :class:`Property <xentica.core.properties.Property>`
            subclass instance.

        """
        cls._unpacks.add(prop)

    def is_declared(cls, prop):
        """
        Check if ``prop`` property is declared.

        :param prop:
            :class:`Property <xentica.core.properties.Property>`
            subclass instance.

        :returns:
            ``True`` if property is declared, ``False`` otherwise.

        """
        return prop in cls._declarations

    def is_unpacked(cls, prop):
        """
        Check if ``prop`` property is unpacked.

        :param prop:
            :class:`Property <xentica.core.properties.Property>`
            subclass instance.

        :returns:
            ``True`` if property is unpacked, ``False`` otherwise.

        """
        return prop in cls._unpacks

    def declare_coords(cls):
        """Mark coordinate variables declared."""
        cls._coords_declared = True

    def define_constant(cls, constant):
        """
        Remember the constant is defined.

        :param constant:
            :class:`Constant <xentica.core.variables.Constant>`
            instance.

        """
        cls.constants[constant.name] = deepcopy(constant)

    def is_constant(cls, constant):
        """
        Check if the constant is defined.

        :param constant:
            :class:`Constant <xentica.core.variables.Constant>`
            instance.

        :returns:
            ``True`` if constant is defined, ``False`` otherwise.

        """
        return constant in cls.constants

    @property
    def coords_declared(cls):
        """
        Check if coordinate variables are declared.

        :returns:
            ``True`` if coordinate variables are declared, ``False``
            otherwise.

        """
        return cls._coords_declared

    def build_defines(cls):
        """
        Generate ``#define`` section for all kernels.

        :returns:
            String with C code with necessary defines.

        """
        defines = ""
        for const in cls.constants.values():
            defines += const.get_define_code()
        return defines

    def build_emit(cls):
        """
        Generate ``emit()`` kernel.

        :returns:
            String with C code for ``emit()`` kernel.

        """
        args = [(cls.ctype, "*fld"), ]
        body = cls._translate_code(cls.emit)
        return cls._elementwise_kernel("emit", args, body)

    def build_absorb(cls):
        """
        Generate ``absorb()`` kernel.

        :returns:
            String with C code for ``absorb()`` kernel.

        """
        args = [(cls.ctype, "*fld"), ("int3", "*col")]
        body = cls._translate_code(cls.absorb, cls.color)
        return cls._elementwise_kernel("absorb", args, body)

    def build_render(cls):
        """
        Generate ``render()`` kernel.

        :returns:
            String with C code for ``render()`` kernel.

        """
        args = cls.renderer.args
        body = cls.renderer.render_code()
        return cls._elementwise_kernel("render", args, body)

    def pack_state(cls, state):
        """
        Pack state structure into raw in-memory representation.

        :returns:
            Integer representing packed state.

        """
        val = 0
        shift = 0
        for name, prop in cls.main.properties.items():
            if name in state:
                val += state[name] << shift
            shift += prop.bit_width
        return val


class GPUKernels:
    """Class incapsulating GPU kernels."""

    def __init__(self, source):
        """
        Initialize GPU kernels.

        :param source:
            CUDA module source code in C.

        """
        cuda_module = SourceModule(source)
        self.emit = cuda_module.get_function("emit")
        self.absorb = cuda_module.get_function("absorb")
        self.render = cuda_module.get_function("render")


class GPUArrays:
    """Class incapsulating GPU arrays."""

    def __init__(self, init_cells, init_colors):
        """
        Initialize GPU arrays.

        :param init_cells:
            NumPy array to put to cells' GPU array.
        :param init_colors:
            NumPy array to put to colors' GPU array.

        """
        self.img = None
        self.colors = gpuarray.to_gpu(init_colors)
        self.cells = gpuarray.to_gpu(init_cells)

    def init_img(self, num_cells):
        """
        Initilatize image for rendering.

        :param num_cells:
            Total number of cells.

        """
        self.img = gpuarray.zeros((num_cells, ), dtype=np.int32)


class GPU:
    """Holder class, incapsulating GPU kernels and arrays."""

    def __init__(self, source, init_cells, init_colors):
        """
        Initialize GPU arrays and kernels.

        :param source:
            CUDA module source code in C.
        :param init_cells:
            NumPy array to put to cells' GPU array.
        :param init_colors:
            NumPy array to put to colors' GPU array.

        """
        self.kernels = GPUKernels(source)
        self.arrays = GPUArrays(init_cells, init_colors)


class CellularAutomaton(metaclass=BSCA):
    """
    Base class for all Xentica models.

    Compiles GPU kernels generated by :class:`BSCA` metaclass,
    initializes necessary GPU arrays and popupates them with the seed.

    After initialization, you can run step-by-step simulation and
    render the field at any moment::

        from xentica import core
        import moire

        class MyCA(core.CellularAutomaton):
            # ...

        class MyExperiment(core.Experiment):
            # ...

        ca = MyCA(MyExperiment)
        ca.set_viewport((320, 200))

        # run CA manually for 100 steps
        for i in range(100):
            ca.step()
        # render current timestep
        frame = ca.render()

        # or run the whole process interactively with Moire
        gui = moire.GUI(runnable=ca)
        gui.run()

    :param experiment_class:
        :class:`Experiment <xentica.core.experiments.Experiment>`
        instance, holding all necessary parameters for the field
        initialization.

    """

    def __init__(self, experiment_class):
        """Initialize kernels, GPU arrays and other stuff."""
        # visuals
        self.frame_buf = np.zeros((3, ), dtype=np.uint8)
        self.width, self.height = 0, 0
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
        for const in self.constants.values():
            source = const.replace_value(source)
        # print(source)
        # build seed
        init_colors = np.zeros((self.cells_num * 3, ), dtype=np.int32)
        cells_total = self.cells_num * (len(self.buffers) + 1) + 1
        self.random = LocalRandom(experiment_class.word)
        experiment_class.seed.random = self.random
        init_cells = np.zeros((cells_total, ), dtype=self.dtype)
        experiment_class.seed.generate(init_cells, self)
        # initialize GPU stuff
        self.gpu = GPU(source, init_cells, init_colors)
        # bridge
        self.bridge = MoireBridge
        self.renderer.setup_actions(self.bridge)
        # lock
        self._lock = threading.Lock()

    def apply_speed(self, dval):
        """
        Change the simulation speed.

        Usable only in conduction with Moire, although you can use the
        ``speed`` value in your custom GUI too.

        :param dval: Delta by which speed is changed.

        """
        self.speed = max(1, (self.speed + dval))

    def toggle_pause(self):
        """
        Toggle ``paused`` flag.

        When paused, the ``step()`` method does nothing.

        """
        self.paused = not self.paused

    def set_viewport(self, size):
        """
        Set viewport (camera) size and initialize GPU array for it.

        :param size: tuple with width and height in pixels.

        """
        self.width, self.height = size
        num_cells = self.width * self.height * 3
        self.gpu.arrays.init_img(num_cells)

    def step(self):
        """
        Perform a single simulation step.

        ``timestep`` attribute will hold the current step number.

        """
        if self.paused:
            return
        # pylint: disable=protected-access
        # This is "hack" to get block/grid sizes, it's vital to us.
        # No way to get it correctly with PyCuda right now.
        block, grid = self.gpu.arrays.cells._block, self.gpu.arrays.cells._grid
        with self._lock:
            self.gpu.kernels.emit(self.gpu.arrays.cells,
                                  np.int32(self.cells_num),
                                  block=block, grid=grid)
            self.gpu.kernels.absorb(self.gpu.arrays.cells,
                                    self.gpu.arrays.colors,
                                    np.int32(self.cells_num),
                                    block=block, grid=grid)
            self.timestep += 1

    def render(self):
        """
        Render the field at the current timestep.

        You must call :meth:`set_viewport` before do any rendering.

        :returns:
            NumPy array of ``np.uint8`` values, ``width * height * 3``
            size. The RGB values are consecutive.

        """
        # pylint: disable=protected-access
        # This is "hack" to get block/grid sizes, it's vital to us.
        # No way to get it correctly with PyCuda right now.
        if self.gpu.arrays.img is None:
            msg = "Viewport is not set, call set_viewport() before rendering."
            raise XenticaException(msg)
        block, grid = self.gpu.arrays.img._block, self.gpu.arrays.img._grid
        with self._lock:
            args = self.renderer.get_args_vals(self)
            args.append(np.int32(self.width * self.height))
            self.gpu.kernels.render(*args, block=block, grid=grid)
            return self.gpu.arrays.img.get().astype(np.uint8)

    def save(self, filename):
        """Save the CA state into ``filename`` file."""
        with open(filename, "wb") as ca_file:
            ca_state = {
                "cells": self.gpu.arrays.cells.get(),
                "colors": self.gpu.arrays.colors.get(),
                "random": self.random,
            }
            pickle.dump(ca_state, ca_file)

    def load(self, filename):
        """Load the CA state from ``filename`` file."""
        with open(filename, "rb") as ca_file:
            ca_state = pickle.load(ca_file)
            self.gpu.arrays.cells = gpuarray.to_gpu(ca_state['cells'])
            self.gpu.arrays.colors = gpuarray.to_gpu(ca_state['colors'])
            self.random.load(ca_state['random'])
