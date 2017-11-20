from xentica.core.topology.mixins import DimensionsMixin


class Border(DimensionsMixin):
    """
    Base class for all types of borders.

    """
    def __init__(self):
        self.topology = None


class TorusBorder(Border):
    supported_dimensions = list(range(1, 100))

    def wrap_coords(self, coord_prefix):
        code = ""
        for i in range(self.dimensions):
            code += "{x}{i} = ({x}{i} + {w}{i}) % {w}{i};\n".format(
                x=coord_prefix, i=i,
                w=self.topology.lattice.width_prefix
            )
        return code


class StaticBorder(Border):
    supported_dimensions = list(range(1, 100))

    def __init__(self, value=0):
        self._value = value

    def off_board_state(self, coord_prefix):
        return str(self._value)
