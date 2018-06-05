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
from collections import OrderedDict

import numpy as np
from cached_property import cached_property

from xentica.core.variables import DeferredExpression

__all__ = ['Property', 'IntegerProperty', 'ContainerProperty', ]


class Property(DeferredExpression):
    """
    Base class for all properties.

    Has a vast set of default functionality already
    implemented. Though, you are free to re-define it all to implement
    really custom behavior.

    """

    def __init__(self):
        """Initialize default attributes."""
        self._bsca = None
        self._types = (
            # (bit_width, numpy_dtype, gpu_c_type)
            (8, np.uint8, 'char'),
            (16, np.uint16, 'short'),
            (32, np.uint32, 'int'),
        )

    @cached_property
    def best_type(self):
        """
        Get type that suits best to store a property.

        :returns:
            tuple representing best type:
            ``(bit_width, numpy_dtype, gpu_c_type)``

        """
        _best_type = self._types[-1]
        for t in self._types:
            type_width = t[0]
            if self.bit_width <= type_width:
                _best_type = t
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

    def calc_bit_width(self):
        """
        Calculate the property's bit width.

        This is the method you most likely need to override. It will
        be called from :meth:`bit_width`.

        :returns:
            Positive integer, calculated property's width in bits.

        """
        return 1  # default, just for consistency

    def set_bsca(self, bsca, buf_num, nbr_num):
        """
        Set up a reference to BSCA instance.

        Do not override this method, it is cruicial to inner framework
        mechanics.

        :param bsca:
            :class:`CellularAutomaton <xentica.core.base.CellularAutomaton>`
            instance.
        :param buf_num:
            Buffer's index, associated to property.
        :param nbr_num:
            Neighbor's index, associated to property.

        """
        self._bsca = bsca
        self._buf_num = buf_num
        self._nbr_num = nbr_num

    def __getattribute__(self, attr):
        """Implement custom logic when property is get as class attribute."""
        obj = object.__getattribute__(self, attr)
        if hasattr(obj, '__get__'):
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
        return DeferredExpression(self.var_name)

    def __set__(self, obj, value):
        """Implement custom logic when property is set as class descriptor."""
        self.declare_once()
        if not hasattr(value, "code"):
            value = DeferredExpression(str(value))
        code = "%s = %s;\n" % (self.var_name, value.code)
        self._bsca.append_code(code)

    @cached_property
    def _mem_cell(self):
        """
        Generate C expression to get cell's state from RAM.

        :returns:
            String with C expression getting the state from memory.

        """
        if self._nbr_num >= 0:
            neighborhood = self._bsca.topology.neighborhood
            return neighborhood.neighbor_state(self._nbr_num,
                                               self._buf_num, "_nx")
        offset = ""
        if self._buf_num > 0:
            offset = " + n * %d" % self._buf_num
        return "fld[i%s]" % offset

    @property
    def _declared(self):
        """Test if the state variable is declared."""
        if self._bsca is None:
            return False
        return self._bsca.is_declared(self)

    @property
    def _coords_declared(self):
        """Test if the coordinates variables are declared."""
        if self._bsca is None:
            return True
        return self._bsca.coords_declared

    def declare_once(self):
        """
        Generate C code to declare a variable holding cell's state.

        You must push the generated code to BSCA via
        ``self._bsca.append_code()``, then declare necessary stuff via
        ``self._bsca.declare()``.

        You should also take care of skipping the whole process if
        things are already declared.

        """
        if self._declared:
            return
        code = "%s %s;\n" % (self.ctype, self.var_name)
        self._bsca.append_code(code)
        self._bsca.declare(self)


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

    def values(self):
        """Iterate over properties, emulating ``dict`` functionality."""
        for p in self._properties.values():
            yield p

    @property
    def _unpacked(self):
        """Test if inner properties are unpacked from memory."""
        if self._bsca is None:
            return False
        return self._bsca.is_unpacked(self)

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

    def set_bsca(self, bsca, buf_num, nbr_num):
        """Propagate BSCA setting to inner properties."""
        self._bsca = bsca
        self._buf_num = buf_num
        self._nbr_num = nbr_num
        for key in self._properties.keys():
            self[key].set_bsca(bsca, buf_num, nbr_num)

    def __get__(self, obj, objtype):
        """Return self reference when getting as class descriptor."""
        return self

    def __set__(self, obj, value):
        """Do nothing when setting as class descriptor."""
        pass

    def __getattribute__(self, attr):
        """Get value from VRAM and unpack it to variables."""
        obj = object.__getattribute__(self, attr)
        if isinstance(obj, Property):
            self.declare_once(self._mem_cell)
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
                self._bsca.deferred_write(self)
            else:
                object.__setattr__(self, attr, val)

    def declare_once(self, init_val=None):
        """
        Do all necessary declarations for inner properties.

        Also, implements the case of off-board neighbor access.

        :param init_val:
             Default value for the property.

        """
        if self._declared:
            return
        code = ""
        if self._nbr_num >= 0:
            neighborhood = self._bsca.topology.neighborhood
            lattice = self._bsca.topology.lattice
            border = self._bsca.topology.border
            dimensions = self._bsca.topology.dimensions

            if not self._coords_declared:
                code += lattice.index_to_coord_code("i", "_x")
                coord_vars = ["_nx%d" % i for i in range(dimensions)]
                code += "int %s;\n" % ", ".join(coord_vars)
                self._bsca.declare_coords()

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
                    neighbor_state=init_val,
                )
                self._bsca.append_code(code)
                self._bsca.declare(self)
                return

        if init_val is None:
            code += "%s %s;\n" % (
                self.ctype, self.var_name
            )
        else:
            code += "%s %s = %s;\n" % (
                self.ctype, self.var_name, init_val
            )
        self._bsca.append_code(code)
        self._bsca.declare(self)

    def _unpack_state(self):
        """Unpack inner properties values from in-memory representation."""
        if self._unpacked:
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
        self._bsca.append_code(code)
        self._bsca.unpack(self)

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
            val = "({val} & {mask})".format(val=prop.var_name, mask=mask)
            if shift > 0:
                val = "({val} << {shift})".format(val=val, shift=shift)
            vals.append(val)
            shift += prop.bit_width
        summed_vals = " + ".join(vals)
        code = "{var} = {val};\n".format(var=self.var_name, val=summed_vals)
        code += "%s = %s;\n" % (self._mem_cell, self.var_name)
        self._bsca.append_code(code)
