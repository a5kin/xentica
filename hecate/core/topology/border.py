from hecate.core.topology.mixins import DimensionsMixin


class Border(DimensionsMixin):
    """
    Base class for all types of borders.

    """
    def __init__(self):
        self.topology = None


class TorusBorder(Border):
    supported_dimensions = [2, ]

    def wrap_coords(self, coord_prefix):
        code = ""
        for i in range(self.dimensions):
            code += "{x}{i} %= {w}{i};\n".format(
                x=coord_prefix, i=i,
                w=self.topology.lattice.width_prefix
            )
        return code
