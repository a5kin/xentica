"""
The collection of classes describing different lattice topologies.

All classes there are intended to be used inside ``Topology`` for
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

from xentica.core.topology.mixins import DimensionsMixin

from xentica.core.mixins import BscaDetectorMixin
from xentica.core.variables import Constant


class Lattice(DimensionsMixin, BscaDetectorMixin, metaclass=abc.ABCMeta):
    """
    Base class for all lattices.

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
        """Define field size conctants in C code."""
        for i in range(self._bsca.topology.dimensions):
            constant = Constant("%s%d" % (self.width_prefix, i),
                                "size[%d]" % i)
            self._bsca.define_constant(constant)

    @abc.abstractmethod
    def index_to_coord_code(self, index_name, coord_prefix):
        """
        Generate C code to obtain coordinates by cell's index.

        This is an abstract method, you must implement it in :class:`Lattice`
        subclasses.

        :param index_name:
            The name of variable containing cell's index.
        :param coord_prefix:
            The prefix for resulting variables, containing coordinates.

        :returns:
            A string with C code, doing all necessary to process
            index variable and store coordinates to variables with
            given prefix.

        """

    @abc.abstractmethod
    def index_to_coord(self, idx, bsca):
        """
        Obtain cell's coordinates by its index, in pure Python.

        This is an abstract method, you must implement it in :class:`Lattice`
        subclasses.

        :param idx:
            Cell's index, a positive integer.
        :param bsca:
            :class:`xentica.core.CellularAutomaton` instance, to access
            field size and number of dimensions.

        :returns:
            Tuple of integer coordinates.

        """

    @abc.abstractmethod
    def coord_to_index_code(self, coord_prefix):
        """
        Generate C code for obtaining cell's index by coordinates.

        This is an abstract method, you must implement it in :class:`Lattice`
        subclasses.

        :param coord_prefix:
            The prefix for variables, containing coordinates.

        :returns:
            A string with C code calculating cell's index. No
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
            A string with C code testing coordinate variables. No
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
        Implement coordinates obtaining by cell's index in C.

        See :meth:`Lattice.index_to_coord_code` for details.

        """
        self._define_constants_once()

        def wrap_format(s):
            return s.format(x=coord_prefix, i=i,
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
        Implement coordinates obtaining by cell's index in Python.

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
        return tuple(coord)

    def coord_to_index_code(self, coord_prefix):
        """
        Implement cell's index obtaining by coordinates in C.

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
        Implement off board cell obtaining in C.

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
