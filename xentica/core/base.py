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
by the number of neighbors, each buffer intended for one of the
neighbors. Then, at each step, the interaction between cells is
performed via buffers in 2-phase emit/absorb process. A more detailed
description of the BSCA principle is available in The Core section of
`The Concept`_ document.


``emit()`` describes the logic of the first phase of BSCA. At this
phase, you should fill the cell's buffers with corresponding values,
depending on the cell's main state and (optionally) on neighbors' main
states. The easiest logic is to just copy the main state to
buffers. It is especially useful when you're intending to emulate
classic CA (like Conway's Life) with BSCA. Write access to the main
state is prohibited there.


``absorb()`` describes the logic of the second phase of BSCA. At this
phase, you should set the cell's main state, depending on neighbors'
buffered states. Write access to buffers is prohibited there.


``color()`` describes how to calculate cell's color from its raw
state. See detailed instructions on it in
:mod:`xentica.core.color_effects`.


The logic of the functions from above will be translated into C code
at the moment of class instance creation. For the further instructions
on how to use the cell's main and buffered states, see
:mod:`xentica.core.properties`, for the instructions on variables and
expressions with them, see :mod:`xentica.core.variables`. Another
helpful thing is meta-parameters, which are described in
:mod:`xentica.core.parameters`.


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
    """The utility class, intended to hold the main and buffered CA state."""

    def __init__(self):
        """Initialize the empty main and buffered states."""
        self.main = ContainerProperty()
        self.buffer = ContainerProperty()


class BSCA(type):
    """
    The meta-class for :class:`CellularAutomaton`.

    Prepares ``main``, ``buffers`` and ``neighbors`` class variables
    being used in ``emit()``, ``absorb()`` and ``color()`` methods.

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
        new_class = super().__new__(mcs, name, bases, attrs)
        mcs._parents = [b for b in bases if isinstance(b, BSCA)]
        if not mcs._parents:
            return new_class

        mcs._prepare_topology(new_class, attrs)

        new_class.main = None
        new_class.buffers = None
        new_class.neighbors = None
        mcs._prepare_properties(new_class, bases, attrs)

        new_class.dtype = new_class.main.dtype
        new_class.ctype = new_class.main.ctype

        new_class.constants = {}

        # set default renderer as needed
        if not hasattr(new_class, 'renderer'):
            new_class.renderer = RendererPlain()

        num_dimensions = new_class.topology.dimensions
        new_class.size = tuple(1 for i in range(num_dimensions))

        return new_class

    def _prepare_topology(cls, attrs):
        """Prepare the topology for future use."""
        if hasattr(cls, 'Topology'):
            attrs['Topology'] = cls.Topology
        cls.topology = attrs.get('Topology')

        if cls.topology is None:
            raise XenticaException("No Topology class declared.")

        for field in cls.mandatory_fields:
            if not hasattr(cls.topology, field):
                msg = "No %s declared in Topology class." % field
                raise XenticaException(msg)

        cls.topology.lattice = deepcopy(cls.topology.lattice)
        cls.topology.neighborhood = deepcopy(cls.topology.neighborhood)
        cls.topology.border = deepcopy(cls.topology.border)
        cls.topology.lattice.dimensions = cls.topology.dimensions
        cls.topology.neighborhood.dimensions = cls.topology.dimensions
        cls.topology.border.dimensions = cls.topology.dimensions
        cls.topology.neighborhood.topology = cls.topology
        cls.topology.border.topology = cls.topology

    def _prepare_properties(cls, bases, attrs):
        """Prepare main/buffers properties."""
        cls.main = ContainerProperty()
        cls.buffers = []
        cls.neighbors = []
        cls.meta = type('MetaParams', (object,), dict())
        num_neighbors = len(cls.topology.neighborhood)
        for i in range(num_neighbors):
            cls.buffers.append(ContainerProperty())
            cls.neighbors.append(CachedNeighbor())
        attrs_items = [base_class.__dict__.items() for base_class in bases]
        attrs_items.append(attrs.items())
        restricted_names = {"main", "buffer"}
        for obj_name, obj in itertools.chain.from_iterable(attrs_items):
            allowed_name = obj_name not in restricted_names
            if isinstance(obj, Property) and allowed_name:
                cls.main[obj_name] = deepcopy(obj)
                vname = "_cell_%s" % obj_name
                cls.main[obj_name].var_name = vname
                for i in range(num_neighbors):
                    buffers = cls.buffers
                    buffers[i][obj_name] = deepcopy(obj)
                    vname = "_bcell_%s%d" % (obj_name, i)
                    buffers[i][obj_name].var_name = vname
                    neighbor = cls.neighbors[i]
                    neighbor.main[obj_name] = deepcopy(obj)
                    vname = "_dcell_%s%d" % (obj_name, i)
                    neighbor.main[obj_name].var_name = vname
                    neighbor.buffer[obj_name] = deepcopy(obj)
                    vname = "_dbcell_%s%d" % (obj_name, i)
                    neighbor.buffer[obj_name].var_name = vname
            elif obj.__class__.__name__ == "Parameter" and allowed_name:
                obj.name = obj_name
                setattr(cls.meta, obj_name, obj)

        cls.main.buf_num = 0
        cls.main.nbr_num = -1
        cls.main.var_name = "_cell"
        for i in range(num_neighbors):
            cls.buffers[i].buf_num = i + 1
            cls.buffers[i].nbr_num = -1
            cls.buffers[i].var_name = "_bcell%i" % i
            cls.neighbors[i].main.buf_num = 0
            cls.neighbors[i].main.nbr_num = i
            cls.neighbors[i].main.var_name = "_dcell%d" % i
            cls.neighbors[i].buffer.buf_num = i + 1
            cls.neighbors[i].buffer.nbr_num = i
            cls.neighbors[i].buffer.var_name = "_dbcell%d" % i


class Translator:
    """The basic functionality for the Python -> CUDA C code translation."""

    def __init__(self):
        self.cuda_source = ""
        self._func_body = ""
        self._deferred_writes = set()
        self._declarations = set()
        self._unpacks = set()
        self._params = {}
        self._emit_params = []
        self._absorb_params = []
        self._render_params = []
        self._coords_declared = False

    @staticmethod
    def _elementwise_kernel(name, args, body):
        """
        Build an elementwise kernel using the template.

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

    def _translate_code(self, *funcs):
        """
        Translate the Python method to C code by execution.

        :param funcs:
            List of functions to translate in a single context.

        :returns:
            String with a generated C code for the elementwise kernel.

        """
        self._func_body = ""
        self._deferred_writes = set()
        self._declarations = set()
        self._unpacks = set()
        self._params = {}
        self._coords_declared = False
        for func in funcs:
            func()
        for prop in self._deferred_writes:
            prop.deferred_write()
        return self._func_body

    def append_code(self, code):
        """Append ``code`` to the kernel's C code."""
        self._func_body += code

    def deferred_write(self, prop):
        """
        Declare property for a deferred write.

        The property will be written into a memory at the end of C
        code generation.

        :param prop:
            :class:`Property <xentica.core.properties.Property>`
            subclass instance.

        """
        self._deferred_writes.add(prop)

    def declare(self, prop):
        """
        Mark the property declared.

        :param prop:
            :class:`Property <xentica.core.properties.Property>`
            subclass instance.

        """
        self._declarations.add(prop)

    def unpack(self, prop):
        """
        Mark the property unpacked.

        :param prop:
            :class:`Property <xentica.core.properties.Property>`
            subclass instance.

        """
        self._unpacks.add(prop)

    def is_declared(self, prop):
        """
        Check if the property is declared.

        :param prop:
            :class:`Property <xentica.core.properties.Property>`
            subclass instance.

        :returns:
            ``True`` if property is declared, ``False`` otherwise.

        """
        return prop in self._declarations

    def is_unpacked(self, prop):
        """
        Check if the property is unpacked.

        :param prop:
            :class:`Property <xentica.core.properties.Property>`
            subclass instance.

        :returns:
            ``True`` if property is unpacked, ``False`` otherwise.

        """
        return prop in self._unpacks

    def declare_coords(self):
        """Mark coordinate variables declared."""
        self._coords_declared = True

    def define_constant(self, constant):
        """
        Remember the constant is defined.

        :param constant:
            :class:`Constant <xentica.core.variables.Constant>`
            instance.

        """
        self.constants[constant.name] = deepcopy(constant)

    def is_constant(self, constant_name):
        """
        Check if the constant is defined.

        :param constant_name:
            The name of the defined constant.

        :returns:
            ``True`` if constant is defined, ``False`` otherwise.

        """
        return constant_name in self.constants

    def define_parameter(self, param):
        """
        Remember the parameter is defined.

        :param param:
            :class:`Parameter <xentica.core.parameters.Parameter>`
            instance.

        """
        self._params[param.name] = param

    def is_parameter(self, param_name):
        """
        Check if the parameter is defined.

        :param param:
            The name of the defined parameter.

        :returns:
            ``True`` if parameter is defined, ``False`` otherwise.

        """
        return param_name in self._params

    @property
    def coords_declared(self):
        """
        Check if coordinate variables are declared.

        :returns:
            ``True`` if coordinate variables are declared, ``False``
            otherwise.

        """
        return self._coords_declared

    def build_source(self):
        """Generate a source code for the GPU kernel."""
        source = self.build_emit()
        source += self.build_absorb()
        source += self.build_render()
        source = self.build_defines() + source
        self.cuda_source = source

    def build_defines(self):
        """
        Generate ``#define`` section for all kernels.

        :returns:
            String with C code with necessary defines.

        """
        defines = ""
        for const in self.constants.values():
            defines += const.get_define_code()
        return defines

    def build_emit(self):
        """
        Generate ``emit()`` kernel.

        :returns:
            String with C code for ``emit()`` kernel.

        """
        args = [(self.ctype, "*fld"), ]
        body = self._translate_code(self.emit)
        self._emit_params = [param for param in self._params.values()]
        args += [(param.ctype, param.name) for param in self._emit_params]
        return self._elementwise_kernel("emit", args, body)

    def build_absorb(self):
        """
        Generate ``absorb()`` kernel.

        :returns:
            String with C code for ``absorb()`` kernel.

        """
        args = [(self.ctype, "*fld"), ("int3", "*col")]
        body = self._translate_code(self.absorb, self.color)
        self._absorb_params = [param for param in self._params.values()]
        args += [(param.ctype, param.name) for param in self._absorb_params]
        return self._elementwise_kernel("absorb", args, body)

    def build_render(self):
        """
        Generate ``render()`` kernel.

        :returns:
            String with C code for ``render()`` kernel.

        """
        args = self.renderer.args
        body = self.renderer.render_code()
        return self._elementwise_kernel("render", args, body)

    def index_to_coord(self, i):
        """
        Wrap ``lattice.index_to_coord`` method.

        :param i: Cell's index.

        """
        return self.topology.lattice.index_to_coord(i, self)

    def pack_state(self, state):
        """
        Pack state structure into raw in-memory representation.

        :returns:
            Integer representing packed state.

        """
        val = 0
        shift = 0
        for name, prop in self.main.properties.items():
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
        cuda_module = SourceModule(source, options=["-w", ])
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
        Initilatize the image for rendering.

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


class CellularAutomaton(Translator, metaclass=BSCA):
    """
    The base class for all Xentica models.

    Generates GPU kernels source code, compiles them, initializes
    necessary GPU arrays and populates them with the seed.

    After initialization, you can run a step-by-step simulation and
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
        """Initialize kernels, GPU arrays, and other stuff."""
        super().__init__()
        # visuals
        self.frame_buf = np.zeros((3, ), dtype=np.uint8)
        self.width, self.height = 0, 0
        # populate attributes from Experiment class
        for attr_name in dir(experiment_class):
            attr = getattr(experiment_class, attr_name)
            if (not callable(attr) and not attr_name.startswith("__")):
                if attr_name == 'seed':
                    continue
                if hasattr(self.meta, attr_name):
                    self.meta.__dict__[attr_name].__set__(self, attr)
                    continue
                setattr(self, attr_name, attr)
        # default simulation values
        self.speed = 1
        self.paused = False
        self.timestep = 0
        # CUDA kernel
        self.cells_num = functools.reduce(operator.mul, self.size)
        self.build_source()
        # print(self.cuda_source)
        # build seed
        init_colors = np.zeros((self.cells_num * 3, ), dtype=np.int32)
        cells_total = self.cells_num * (len(self.buffers) + 1) + 1
        self.random = LocalRandom(experiment_class.word)
        experiment_class.seed.random = self.random
        init_cells = np.zeros((cells_total, ), dtype=self.dtype)
        experiment_class.seed.generate(init_cells, self)
        # initialize GPU stuff
        self.gpu = GPU(self.cuda_source, init_cells, init_colors)
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
        Set the viewport (camera) size and initialize GPU array for it.

        :param size: tuple with the width and height in pixels.

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
            args = [param.dtype(param.value) for param in self._emit_params]
            self.gpu.kernels.emit(self.gpu.arrays.cells,
                                  *args,
                                  np.int32(self.cells_num),
                                  block=block, grid=grid)
            args = [param.dtype(param.value) for param in self._absorb_params]
            self.gpu.kernels.absorb(self.gpu.arrays.cells,
                                    self.gpu.arrays.colors,
                                    *args,
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
            args += [param.dtype(param.value) for param in self._render_params]
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
