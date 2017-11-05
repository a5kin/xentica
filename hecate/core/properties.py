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
        self._declare_once(self._mem_cell)
        return DeferredExpression(self.var_name)

    def __set__(self, obj, value):
        self._declare_once()
        code = "%s = %s;\n" % (self.var_name, value.code)
        self._bsca.append_code(code)
        self._bsca.deferred_write(self)

    @cached_property
    def _mem_cell(self):
        offset = ""
        if self._buf_num > 0:
            offset = " + n * %d" % self._buf_num
        return "fld[i%s]" % offset

    @property
    def _declared(self):
        if self._bsca is None:
            return False
        if self.var_name[:7] == "_dbcell":
            return True
        return self._bsca.is_declared(self)

    def _declare_once(self, init_val=None):
        if not self._declared:
            c = "%s %s;\n" % (
                self.ctype, self.var_name
            )
            if init_val is not None:
                c = "%s %s = %s;\n" % (
                    self.ctype, self.var_name, init_val
                )
            self._bsca.append_code(c)
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
