"""
The collection of classes to describe properties of CA models.

.. warning::
    Do not confuse with Python properties.

Xentica properties are declaring as class variables and helping you to
organize CA state into complex structures.

Each :class:`CellularAutomaton <xentica.core.base.CellularAutomaton>`
instance should have at least one property declared. The property name
is up to you. If your model has just one value for state (like in most
classic CA), the best practice is to call it ``state`` as follows::

    from xentica import core

    class MyCA(core.CellularAutomaton):
        state = core.IntegerProperty(max_val=1)
        # ...

Then, you can use it in expressions of ``emit()``, ``absorb()`` and
``color()`` functions as:

``self.main.state``
    to get and set main state;

``self.buffers[i].state``
    to get and set i-th buffered state;

``self.neighbors[i].buffer.state``
    to get and set i-th neighbor buffered state.

Xentica will take care of all other things, like packing CA properties
into binary representation and back, storing and getting corresponding
values from VRAM, etc.

Most of properties will return
:class:`DeferredExpression <xentica.core.variables.DeferredExpression>`
on access, so you can use them safely in mixed expressions::

    self.buffers[i].state = self.main.state + 1

"""
import math
import abc
import inspect
from collections import OrderedDict

import numpy as np
from cached_property import cached_property

from xentica.core.expressions import DeferredExpression
from xentica.core.exceptions import XenticaException
from xentica.core.mixins import BscaDetectorMixin
from xentica.tools import xmath

__all__ = ['Property', 'IntegerProperty', 'ContainerProperty', ]


class Property(DeferredExpression, BscaDetectorMixin):
    """
    Base class for all properties.

    Has a vast set of default functionality already
    implemented. Though, you are free to re-define it all to implement
    really custom behavior.

    """

    def __init__(self):
        """Initialize default attributes."""
        self._types = (
            # (bit_width, numpy_dtype, gpu_c_type)
            (8, np.uint8, 'char'),
            (16, np.uint16, 'short'),
            (32, np.uint32, 'int'),
            (64, np.uint64, 'long'),
        )
        self._buf_num, self._nbr_num = 1, 1
        super(Property, self).__init__()

    @cached_property
    def best_type(self):
        """
        Get type that suits best to store a property.

        :returns:
            tuple representing best type:
            ``(bit_width, numpy_dtype, gpu_c_type)``

        """
        _best_type = self._types[-1]
        for _type in self._types:
            type_width = _type[0]
            if self.bit_width <= type_width:
                _best_type = _type
                break
        return _best_type

    @cached_property
    def dtype(self):
        """
        Get NumPy dtype, based on result of :meth:`best_type`.

        :returns:
            NumPy dtype that suits best to store a property.

        """
        return self.best_type[1]

    @cached_property
    def ctype(self):
        """
        Get C type, based on result of :meth:`best_type`.

        :returns:
            C type that suits best to store a property.

        """
        return 'unsigned ' + self.best_type[2]

    @property
    def _num_neighbors(self):
        """Get number of neighbors for a cell in the current model."""
        return len(self.bsca.buffers)

    @cached_property
    def bit_width(self):
        """
        Get the number of bits necessary to store a property.

        :returns:
            Positive integer, a property's bit width.

        """
        return self.calc_bit_width()

    @cached_property
    def width(self):
        """
        Get the number of memory cells to store a property.

        In example, if ``ctype == "int"`` and ``bit_width == 64``, you
        need 2 memory cells.

        :returns:
            Positive integer, a property's width.

        """
        type_width = self.best_type[0]
        return int(math.ceil(self.bit_width / type_width))

    @abc.abstractmethod
    def calc_bit_width(self):
        """
        Calculate the property's bit width.

        This is the method you most likely need to override. It will
        be called from :meth:`bit_width`.

        :returns:
            Positive integer, calculated property's width in bits.

        """
        return 1  # default, just for consistency

    @property
    def buf_num(self):
        """Get buffer's index, associated to property."""
        return self._buf_num

    @buf_num.setter
    def buf_num(self, val):
        """Set buffer's index, associated to property."""
        self._buf_num = val

    @property
    def nbr_num(self):
        """Get neighbor's index, associated to property."""
        return self._nbr_num

    @nbr_num.setter
    def nbr_num(self, val):
        """Set neighbor's index, associated to property."""
        self._nbr_num = val

    def __getattribute__(self, attr):
        """Implement custom logic when property is get as class attribute."""
        obj = object.__getattribute__(self, attr)
        if hasattr(obj, '__get__') and attr != "__class__":
            return obj.__get__(self, type(self))
        return obj

    def __setattr__(self, attr, val):
        """Implement custom logic when property is set as class attribute."""
        try:
            obj = object.__getattribute__(self, attr)
        except AttributeError:
            object.__setattr__(self, attr, val)
        else:
            if hasattr(obj, '__set__'):
                obj.__set__(self, val)
            else:
                object.__setattr__(self, attr, val)

    def __get__(self, obj, objtype):
        """Implement custom logic when property is get as class descriptor."""
        self.declare_once()
        return self

    def __set__(self, obj, value):
        """Implement custom logic when property is set as class descriptor."""
        self.declare_once()
        code = "%s = %s;\n" % (self, value)
        self.bsca.append_code(code)

    @cached_property
    def _mem_cell(self):
        """
        Generate C expression to get cell's state from RAM.

        :returns:
            String with C expression getting the state from memory.

        """
        if self._nbr_num >= 0:
            neighborhood = self.bsca.topology.neighborhood
            return neighborhood.neighbor_state(self._nbr_num,
                                               self._buf_num, "_nx")
        offset = ""
        if self._buf_num > 0:
            offset = " + n * %d" % self._buf_num
        return "fld[i%s]" % offset

    @property
    def declared(self):
        """Test if the state variable is declared."""
        return self.bsca.is_declared(self)

    @property
    def coords_declared(self):
        """Test if the coordinates variables are declared."""
        return self.bsca.coords_declared

    def declare_once(self):
        """
        Generate C code to declare a variable holding cell's state.

        You must push the generated code to BSCA via
        ``self.bsca.append_code()``, then declare necessary stuff via
        ``self.bsca.declare()``.

        You should also take care of skipping the whole process if
        things are already declared.

        """
        if self.declared:
            return
        code = "%s %s;\n" % (self.ctype, self.var_name)
        self.bsca.append_code(code)
        self.bsca.declare(self)

    def __str__(self):
        """Return a variable name to use in mixed expressions."""
        return self.var_name


class IntegerProperty(Property):
    """
    Most generic property for you to use.

    It is just a positive integer with upper limit of ``max_val``.

    """

    def __init__(self, max_val):
        """Initialize class specific attributes."""
        self.max_val = max_val
        self._buf_num = 0
        super(IntegerProperty, self).__init__()

    def calc_bit_width(self):
        """Calculate bit width, based on ``max_val``."""
        return int(math.log2(self.max_val)) + 1


class ContainerProperty(Property):
    """
    A property acting as a holder for other properties.

    Currently is used only for inner framework mechanics, in
    particular, to hold, pack and unpack all top-level properties.

    It will be enhanced in future versions, and give you the
    ability to implement nested properties structures.

    .. warning::
        Right now, direct use of this class is prohibited.

    """

    def __init__(self):
        """Initialize ``OrderedDict`` to hold other properties."""
        super(ContainerProperty, self).__init__()
        self._properties = OrderedDict()
        self.init_val = None

    @property
    def properties(self):
        """Get inner properties."""
        return self._properties

    def values(self):
        """Iterate over properties, emulating ``dict`` functionality."""
        for prop in self._properties.values():
            yield prop

    @property
    def unpacked(self):
        """Test if inner properties are unpacked from memory."""
        return self.bsca.is_unpacked(self)

    def __getitem__(self, key):
        """Get property by key, emulating ``dict`` functionality."""
        return self._properties[key]

    def __setitem__(self, key, val):
        """Set property by key, emulating ``dict`` functionality."""
        self._properties[key] = val
        object.__setattr__(self, key, val)

    def calc_bit_width(self):
        """Calculate bit width as sum of inner properties' bit widths."""
        return sum([p.bit_width for p in self._properties.values()])

    @Property.buf_num.setter
    def buf_num(self, val):
        """Set buffer's index, and propagade to inner properties."""
        self._buf_num = val
        for key in self._properties.keys():
            self[key].buf_num = val

    @Property.nbr_num.setter
    def nbr_num(self, val):
        """Set neighbor's index, and propagade to inner properties."""
        self._nbr_num = val
        for key in self._properties.keys():
            self[key].nbr_num = val

    def __get__(self, obj, objtype):
        """Return self reference when getting as class descriptor."""
        return self

    def __set__(self, obj, value):
        """Do nothing when setting as class descriptor."""

    def __getattribute__(self, attr):
        """Get value from VRAM and unpack it to variables."""
        obj = object.__getattribute__(self, attr)
        if isinstance(obj, Property):
            self.init_val = self._mem_cell
            self.declare_once()
            self._unpack_state()
            return obj.__get__(self, type(self))
        return obj

    def __setattr__(self, attr, val):
        """Declare resulting variable and defer the write memory access."""
        try:
            obj = object.__getattribute__(self, attr)
        except AttributeError:
            object.__setattr__(self, attr, val)
        else:
            if isinstance(obj, Property):
                obj.declare_once()
                obj.__set__(self, val)
                self.declare_once()
                self.bsca.deferred_write(self)
                self.bsca.unpack(self)
            else:
                object.__setattr__(self, attr, val)

    def declare_once(self):
        """
        Do all necessary declarations for inner properties.

        Also, implements the case of off-board neighbor access.

        :param init_val:
             Default value for the property.

        """
        if self.declared:
            return
        code = ""
        if self._nbr_num >= 0:
            neighborhood = self.bsca.topology.neighborhood
            lattice = self.bsca.topology.lattice
            border = self.bsca.topology.border
            dimensions = self.bsca.topology.dimensions

            if not self.coords_declared:
                code += lattice.index_to_coord_code("i", "_x")
                coord_vars = ["_nx%d" % i for i in range(dimensions)]
                code += "int %s;\n" % ", ".join(coord_vars)
                self.bsca.declare_coords()

            code += neighborhood.neighbor_coords(self._nbr_num, "_x", "_nx")
            is_cell_off_board = lattice.is_off_board_code("_nx")
            if hasattr(border, "wrap_coords"):
                code += """
                    if ({is_cell_off_board}) {{
                        {wrap_coords}
                    }}
                """.format(
                    is_cell_off_board=is_cell_off_board,
                    wrap_coords=border.wrap_coords("_nx"),
                )
            else:
                code += """
                    {type} {var};
                    if ({is_cell_off_board}) {{
                        {var} = {off_board_cell};
                    }} else {{
                        {var} = {neighbor_state};
                    }}
                """.format(
                    type=self.ctype, var=self.var_name,
                    is_cell_off_board=is_cell_off_board,
                    off_board_cell=border.off_board_state("_nx"),
                    neighbor_state=self.init_val,
                )
                self.bsca.append_code(code)
                self.bsca.declare(self)
                return

        if self.init_val is None:
            code += "%s %s;\n" % (
                self.ctype, self.var_name
            )
        else:
            code += "%s %s = %s;\n" % (
                self.ctype, self.var_name, self.init_val
            )
        self.bsca.append_code(code)
        self.bsca.declare(self)

    def _unpack_state(self):
        """Unpack inner properties values from in-memory representation."""
        if self.unpacked:
            return
        code = ""
        shift = 0
        for prop in self._properties.values():
            prop.declare_once()
            val = self.var_name
            if shift > 0:
                val += " >> %d" % shift
            mask = 2 ** prop.bit_width - 1
            val = "({val}) & {mask}".format(val=val, mask=mask)
            code += "{var} = {val};\n".format(var=prop.var_name, val=val)
            shift += prop.bit_width
        self.bsca.append_code(code)
        self.bsca.unpack(self)

    def deferred_write(self):
        """
        Pack state and write its value to VRAM.

        This method is called from ``BSCA`` at the end of kernel processing.

        """
        shift = 0
        vals = []
        for prop in self._properties.values():
            prop.declare_once()
            mask = 2 ** prop.bit_width - 1
            val = "(({ctype}) {val} & {mask})".format(
                ctype=self.ctype,
                val=prop.var_name,
                mask=mask
            )
            if shift > 0:
                val = "({val} << {shift})".format(val=val, shift=shift)
            vals.append(val)
            shift += prop.bit_width
        summed_vals = " + ".join(vals)
        code = "{var} = {val};\n".format(var=self.var_name, val=summed_vals)
        code += "%s = %s;\n" % (self._mem_cell, self.var_name)
        self.bsca.append_code(code)


class TotalisticRuleProperty(Property):
    """A specific property implementing totalistic rule."""

    def __init__(self, outer=False):
        """Initialize the rule."""
        self._buf_num = 0
        self._outer = outer
        super(TotalisticRuleProperty, self).__init__()

    @property
    def _genome_mask(self):
        """Get the sanity mask for foolproof genome operations."""
        return 2 ** (self._num_neighbors + 1) - 2

    def calc_bit_width(self):
        """Calculate bit width, based on number of neighbors."""
        return (self._num_neighbors + 1) * 2

    def is_sustained(self, num_neighbors):
        """
        Determine if cell is living or intended to death.

        :param num_neighbors: Number of neighbors to test over.

        :returns: ``DeferredExpression`` to calculate the bool value.

        """
        if not self._outer:
            msg = "Can not get sustained flag from pure totalistic rule."
            raise XenticaException(msg)
        mask = (self._genome_mask << (self._num_neighbors + 1))
        return ((self & mask) >> (num_neighbors + self._num_neighbors + 1)) & 1

    def is_born(self, num_neighbors):
        """
        Determine if cell could be born.

        :param num_neighbors: Number of neighbors to test over.

        :returns: ``DeferredExpression`` to calculate the bool value.

        """
        if not self._outer:
            msg = "Can not get born flag from pure totalistic rule."
            raise XenticaException(msg)
        return ((self & self._genome_mask) >> num_neighbors) & 1

    def __get__(self, obj, objtype):
        """Declare and return self when get as class descriptor."""
        self.declare_once()
        return self

    def next_val(self, cur_val, num_neighbors):
        """
        Calculate cell's value at the next step.

        :param cur_val: Current cell's value.
        :param num_neighbors: Number of neighbors to test over.

        :returns: ``DeferredExpression`` to calculate the bool value.

        """
        if not self._outer:
            return (self >> (num_neighbors + cur_val)) & 1
        alive_shift = cur_val * (self._num_neighbors + 1)
        return (self >> (num_neighbors + alive_shift)) & 1


class RandomProperty(Property):
    """A property yielding random value each time."""

    def __init__(self):
        self._buf_num = 0
        super(RandomProperty, self).__init__()

    def __get__(self, obj, objtype):
        """Implement custom logic when property is get as class descriptor."""
        self.declare_once()
        self._get_next()
        return self

    def calc_bit_width(self):
        """Really dirty 16-bit PRNG."""
        return 16

    def _get_next(self):
        """Generate and return next value for RNG stream."""
        val_int = xmath.int(DeferredExpression(self.var_name))
        val = ((val_int * 58321 + 11113)) % 65535
        self.__set__(self, val)
        container = inspect.currentframe().f_back.f_locals['obj']
        self.bsca.deferred_write(container)

    @property
    def uniform(self):
        """Get random value from uniform distribution."""
        return xmath.float(DeferredExpression(self.var_name)) / 65535
