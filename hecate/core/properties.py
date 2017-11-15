import math
from collections import OrderedDict

import numpy as np
from cached_property import cached_property

from hecate.core.variables import DeferredExpression


class Property(DeferredExpression):
    """
    Base class for all properties.

    """
    def __init__(self):
        self._bsca = None
        self._types = (
            # (bit_width, numpy_dtype, gpu_c_type)
            (8, np.uint8, 'char'),
            (16, np.uint16, 'short'),
            (32, np.uint32, 'int'),
        )

    @cached_property
    def best_type(self):
        _best_type = self._types[-1]
        for t in self._types:
            type_width = t[0]
            if self.bit_width <= type_width:
                _best_type = t
                break
        return _best_type

    @cached_property
    def dtype(self):
        return self.best_type[1]

    @cached_property
    def ctype(self):
        return 'unsigned ' + self.best_type[2]

    @cached_property
    def bit_width(self):
        return self.calc_bit_width()

    @cached_property
    def width(self):
        type_width = self.best_type[0]
        return int(math.ceil(self.bit_width / type_width))

    def calc_bit_width(self):
        return 1  # default, just for consistency

    def set_bsca(self, bsca, buf_num, nbr_num):
        self._bsca = bsca
        self._buf_num = buf_num
        self._nbr_num = nbr_num

    def __getattribute__(self, attr):
        obj = object.__getattribute__(self, attr)
        if hasattr(obj, '__get__'):
            return obj.__get__(self, type(self))
        return obj

    def __setattr__(self, attr, val):
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
        self.declare_once()
        return DeferredExpression(self.var_name)

    def __set__(self, obj, value):
        self.declare_once()
        code = "%s = %s;\n" % (self.var_name, value.code)
        self._bsca.append_code(code)

    @cached_property
    def _mem_cell(self):
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
        if self._bsca is None:
            return False
        return self._bsca.is_declared(self)

    @property
    def _coords_declared(self):
        if self._bsca is None:
            return True
        return self._bsca.coords_declared

    def declare_once(self, init_val=None):
        if self._declared:
            return
        code = "%s %s;\n" % (self.ctype, self.var_name)
        self._bsca.append_code(code)
        self._bsca.declare(self)


class IntegerProperty(Property):

    def __init__(self, max_val):
        self.max_val = max_val
        self._buf_num = 0
        super(IntegerProperty, self).__init__()

    def calc_bit_width(self):
        return int(math.log2(self.max_val)) + 1


class ContainerProperty(Property):

    def __init__(self):
        super(ContainerProperty, self).__init__()
        self._properties = OrderedDict()

    def items(self):
        for p in self._properties.values():
            yield p

    @property
    def _unpacked(self):
        if self._bsca is None:
            return False
        return self._bsca.is_unpacked(self)

    def __getitem__(self, key):
        return self._properties[key]

    def __setitem__(self, key, val):
        self._properties[key] = val
        object.__setattr__(self, key, val)

    def calc_bit_width(self):
        return sum([p.bit_width for p in self._properties.values()])

    def set_bsca(self, bsca, buf_num, nbr_num):
        self._bsca = bsca
        self._buf_num = buf_num
        self._nbr_num = nbr_num
        for key in self._properties.keys():
            self[key].set_bsca(bsca, buf_num, nbr_num)

    def __get__(self, obj, objtype):
        return self

    def __set__(self, obj, value):
        pass

    def __getattribute__(self, attr):
        obj = object.__getattribute__(self, attr)
        if isinstance(obj, Property):
            self.declare_once(self._mem_cell)
            self._unpack_state()
            return obj.__get__(self, type(self))
        return obj

    def __setattr__(self, attr, val):
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
        Pack state and write its value to VRAM

        """
        shift = 0
        vals = []
        for prop in self._properties.values():
            prop.declare_once()
            val = prop.var_name
            if shift > 0:
                val += " << %d" % shift
            mask = 2 ** prop.bit_width - 1
            vals.append("(({val}) & {mask})".format(val=val, mask=mask))
            shift += prop.bit_width
        summed_vals = " + ".join(vals)
        code = "{var} = {val};\n".format(var=self.var_name, val=summed_vals)
        code += "%s = %s;\n" % (self._mem_cell, self.var_name)
        self._bsca.append_code(code)
