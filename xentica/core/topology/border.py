"""
The collection of classes describing different types of field borders.

All classes there are intended to be used inside ``Topology`` for
``border`` class variable definition. They are also available via
:mod:`xentica.core` shortcut. The example::

    from xentica.core import CellularAutomaton, TorusBorder

    class MyCA(CellularAutomaton)
        class Topology:
            border = TorusBorder()
            # ...
        # ...

"""
from xentica.core.topology.mixins import DimensionsMixin


class Border(DimensionsMixin):
    """
    Base class for all types of borders.

    You should not inherit your borders directly from this class, use
    either :class:`WrappedBorder` or :class:`GeneratedBorder` base
    subclasses for this.

    """

    def __init__(self):
        """Initialize common things for all borders."""
        self.topology = None


class TorusBorder(Border):
    """
    Wraps the entire field into N-torus manifold.

    This is the most common type of border, allowing you to generate
    seamless tiles for wallpapers.

    """

    #: Any number of dimentions is supported, 100 is just to limit your
    #: hyperspatial hunger.
    supported_dimensions = list(range(1, 100))

    def wrap_coords(self, coord_prefix):
        """
        Impement coordinates wrapping to torus.

        See :meth:`WrappedBorder.wrap_coords` for details.

        """
        code = ""
        for i in range(self.dimensions):
            code += "{x}{i} = ({x}{i} + {w}{i}) % {w}{i};\n".format(
                x=coord_prefix, i=i,
                w=self.topology.lattice.width_prefix
            )
        return code


class StaticBorder(Border):
    """
    Generates a static value for every off-board cell.

    This is acting like your field is surrounded by cells with the
    same pre-defined state.

    The default is just an empty (zero) state.

    """

    supported_dimensions = list(range(1, 100))

    def __init__(self, value=0):
        """
        Store the static value.

        :param value:
            A static value to be used for every off-board cell.

        """
        self._value = value

    def off_board_state(self, coord_prefix):
        """
        Impement off-board cells' values obtaining.

        See :meth:`GeneratedBorder.off_board_state` for details.

        """
        return str(self._value)
