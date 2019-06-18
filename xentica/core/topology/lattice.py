"""
The collection of classes describing different lattice topologies.

All classes there are intended for use inside ``Topology`` for
``lattice`` class variable definition. They are also available via
:mod:`xentica.core` shortcut. The example::

    from xentica.core import CellularAutomaton, OrthogonalLattice

    class MyCA(CellularAutomaton):
        class Topology:
            lattice = OrthogonalLattice()
            # ...
        # ...

"""
import abc

from xentica.core.mixins import DimensionsMixin, BscaDetectorMixin
from xentica.core.variables import Constant
from xentica.core.exceptions import XenticaException

__all__ = ['Lattice', 'OrthogonalLattice', ]


class Lattice(DimensionsMixin, BscaDetectorMixin, metaclass=abc.ABCMeta):
    """
    The base class for all lattices.

    For correct behavior, lattice classes should be inherited from
    this class. You should also implement the following functions:

    - :meth:`index_to_coord_code`

    - :meth:`index_to_coord`

    - :meth:`coord_to_index_code`

    - :meth:`is_off_board_code`

    See the detailed description below.

    """

    #: The prefix to be used in C code for field size constants.
    width_prefix = "_w"

    def _define_constants_once(self):
        """Define field size constants in the C code."""
        num_dimensions = self.bsca.topology.dimensions
        for i in range(num_dimensions):
            if not hasattr(self.bsca, "size") or i >= len(self.bsca.size):
                msg = "Wrong field's dimensionality ({} instead of {})."
                msg = msg.format(len(self.bsca.size), num_dimensions)
                raise XenticaException(msg)
            size = self.bsca.size[i]
            constant = Constant("%s%d" % (self.width_prefix, i), size)
            self.bsca.define_constant(constant)

    @abc.abstractmethod
    def index_to_coord_code(self, index_name, coord_prefix):
        """
        Generate C code to obtain coordinates by the cell's index.

        This is an abstract method, you must implement it in :class:`Lattice`
        subclasses.

        :param index_name:
            The name of a variable containing the cell's index.
        :param coord_prefix:
            The prefix for resulting variables, containing coordinates.

        :returns:
            A string with the C code, doing all necessary to process
            the index variable and store coordinates to variables with
            the given prefix.

        """

    @abc.abstractmethod
    def index_to_coord(self, idx, bsca):
        """
        Obtain the cell's coordinates by its index, in pure Python.

        This is an abstract method, you must implement it in :class:`Lattice`
        subclasses.

        :param idx:
            Cell's index, a positive integer, or a NumPy array of indices.
        :param bsca:
            :class:`xentica.core.CellularAutomaton` instance, to access
            the field's size and number of dimensions.

        :returns:
            Tuple of integer coordinates, or NumPy arrays of coords
            per each axis.

        """

    @abc.abstractmethod
    def coord_to_index_code(self, coord_prefix):
        """
        Generate C code for obtaining the cell's index by coordinates.

        This is an abstract method, you must implement it in :class:`Lattice`
        subclasses.

        :param coord_prefix:
            The prefix for variables, containing coordinates.

        :returns:
            A string with the C code calculating cell's index. No
            assignment, only a valid expression needed.

        """

    @abc.abstractmethod
    def is_off_board_code(self, coord_prefix):
        """
        Generate C code to test if the cell's coordinates are off board.

        This is an abstract method, you must implement it in :class:`Lattice`
        subclasses.

        :param coord_prefix:
            The prefix for variables, containing coordinates.

        :returns:
            A string with the C code testing coordinates' variables. No
            assignment, only a valid expression with boolean result needed.

        """


class OrthogonalLattice(Lattice):
    """
    N-dimensional orthogonal lattice.

    Points are all possible positive integer coordinates.

    """

    #: Overridden value for supported dimensions.
    supported_dimensions = list(range(1, 100))

    def index_to_coord_code(self, index_name, coord_prefix):
        """
        Generate C code for obtaining the cell's index by coordinates.

        See :meth:`Lattice.index_to_coord_code` for details.

        """
        self._define_constants_once()
        i = 0

        def wrap_format(text):
            """Format helper."""
            return text.format(x=coord_prefix, i=i,
                               index=index_name, w=self.width_prefix)

        for i in range(self.dimensions):
            if i == 0:
                code = wrap_format("int _{index} = {index};\n")
                index_name = "_" + index_name
            if i < self.dimensions - 1:
                code += wrap_format("int {x}{i} = {index} % {w}{i};\n")
                code += wrap_format("{index} /= {w}{i};\n")
            else:
                code += wrap_format("int {x}{i} = {index};\n")
        return code

    def index_to_coord(self, idx, bsca):
        """
        Obtain the cell's coordinates by its index, in pure Python.

        See :meth:`Lattice.index_to_coord` for details.

        """
        coord = []
        for i in range(bsca.topology.dimensions):
            if i < self.dimensions - 1:
                x_i = idx % bsca.size[i]
                idx //= bsca.size[i]
            else:
                x_i = idx
            coord.append(x_i)
        return coord

    def coord_to_index_code(self, coord_prefix):
        """
        Generate C code for obtaining the cell's index by coordinates.

        See :meth:`Lattice.coord_to_index_code` for details.

        """
        self._define_constants_once()

        summands = []
        for i in range(self.dimensions):
            summand = coord_prefix + str(i)
            for j in range(i):
                summand = self.width_prefix + str(j) + " * " + summand
            summands.append(summand)
        return " + ".join(summands)

    def is_off_board_code(self, coord_prefix):
        """
        Generate C code to test if the cell's coordinates are off board.

        See :meth:`Lattice.is_off_board_code` for details.

        """
        self._define_constants_once()

        conditions = []
        for i in range(self.dimensions):
            condition = "{x}{i} < 0 || {x}{i} >= {w}{i}".format(
                x=coord_prefix, i=i, w=self.width_prefix
            )
            conditions.append(condition)
        return " || ".join(conditions)
