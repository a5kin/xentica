import math

import numpy as np


class Property:
    """
    Base class for all properties.

    """
    def __init__(self):
        self._dtype = None
        self._ctype = None
        self._bit_width = None
        self._width = None

    @property
    def dtype(self):
        if self._dtype is not None:
            return self._dtype
        self.bit_width

    @property
    def ctype(self):
        if self._ctype is not None:
            return self._ctype

    @property
    def bit_width(self):
        if self._bit_width is None:
            self._bit_width = self.calc_bit_width()
        return self._bit_width

    @property
    def width(self):
        if self._width is not None:
            return self._width

    def calc_bit_width(self):
        return 1  # default, just for consistency


class IntegerProperty(Property):

    def __init__(self, max_val):
        self.max_val = max_val

    def calc_bit_width(self):
        return int(math.log2(self.max_val)) + 1


class ContainerProperty(Property):

    @property
    def dtype(self):
        # hardcoded
        return np.uint8

    @property
    def ctype(self):
        # hardcoded
        return "unsigned char"
