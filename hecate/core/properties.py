import math


class Property:
    """
    Base class for all properties.

    """


class IntegerProperty(Property):

    def __init__(self, max_val):
        self.max_val = max_val
        self.bit_width = int(math.log2(max_val)) + 1


class ContainerProperty(Property):
    pass
