import math

import numpy as np


class Property:
    """
    Base class for all properties.

    """


class IntegerProperty(Property):

    def __init__(self, max_val):
        self.max_val = max_val
        self.bit_width = int(math.log2(max_val)) + 1


class ContainerProperty(Property):

    @property
    def dtype(self):
        # hardcoded
        return np.uint8

    @property
    def cudatype(self):
        # hardcoded
        return "unsigned char"
