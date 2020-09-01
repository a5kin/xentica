"""
The collection of classes describing different neighborhood topologies.

All classes there are intended for use inside ``Topology`` for
``neighborhood`` class variable definition. They are also available via
:mod:`xentica.core` shortcut. The example::

    from xentica.core import CellularAutomaton, MooreNeighborhood

    class MyCA(CellularAutomaton):
        class Topology:
            neighborhood = MooreNeighborhood()
            # ...
        # ...

"""
import itertools
import abc

from xentica.core.mixins import DimensionsMixin

__all__ = [
    'Neighborhood', 'OrthogonalNeighborhood',
    'MooreNeighborhood', 'VonNeumannNeighborhood',
]


class Neighborhood(DimensionsMixin):
    """
    The base class for all neighborhood topologies.

    For correct behavior, neighborhood classes should be inherited from
    this class. You should also implement the following functions:

    - :meth:`neighbor_coords`

    - :meth:`neighbor_state`

    See the detailed description below.

    """

    def __init__(self):
        """Initialize main attributes."""
        #: Number of neighbors, you must re-define it in sub-classes.
        self.num_neighbors = None
        #: A reference to ``Topology`` holder class, will be set in
        #: ``BSCA`` metaclass.
        self.topology = None
        self._delta2str = {-1: " - 1", 0: "", 1: " + 1"}
        self._neighbor_deltas = []
        self.rev_index_map = []
        super().__init__()

    def __len__(self):
        """Return the number of neighbors for a single cell."""
        return self.num_neighbors or 0

    @abc.abstractmethod
    def neighbor_coords(self, index, coord_prefix, neighbor_prefix):
        """
        Generate the C code to obtain neighbor coordinates by its index.

        This is an abstract method, you must implement it in
        :class:`Neighborhood` subclasses.

        :param index:
            Neighbor's index, a non-negative integer less than the number of
            neighbors.
        :param coord_prefix:
            The prefix for variables containing main cell's coordinates.
        :param neighbor_prefix:
            The prefix for resulting variables containing neighbor coordinates.

        :returns:
            A string with the C code doing all necessary to get neighbor's
            state from the RAM. No assignment, only a valid expression
            needed.

        """

    @abc.abstractmethod
    def neighbor_state(self, neighbor_index, state_index, coord_prefix):
        """
        Generate the C code to obtain a neighbor's state by its index.

        This is an abstract method, you must implement it in
        :class:`Neighborhood` subclasses.

        :param neighbor_index:
            Neighbor's index, a non-negative integer less than the number of
            neighbors.
        :param state_index:
            State's index, a non-negative integer less than the number of
            neighbors for buffered states or -1 for main state.
        :param coord_prefix:
            The prefix for variables containing neighbor coordinates.

        :returns:
            A string with the C code doing all necessary to process
            neighbor's coordinates and store them to neighbor's
            coordinates variables.

        """


class OrthogonalNeighborhood(Neighborhood):
    """
    The base class for neighborhoods on an orthogonal lattice.

    It is implementing all necessary :class:`Neighborhood` abstract
    methods, the only thing you should override is :meth:`dimensions`
    setter. In :meth:`dimensions`, you should correctly set
    ``num_neighbors`` and ``_neighbor_deltas`` attributes.

    """

    #: Any number of dimentions is supported, 100 is just to limit your
    #: hyperspatial hunger.
    supported_dimensions = list(range(1, 100))

    def neighbor_coords(self, index, coord_prefix, neighbor_prefix):
        """
        Generate the C code to obtain neighbor coordinates by its index.

        See :meth:`Neighborhood.neighbor_coords` for details.

        """
        code = ""
        for i in range(self.dimensions):
            code += "{neighbor}{i} = {coord}{i}{delta};\n".format(
                neighbor=neighbor_prefix, i=i,
                coord=coord_prefix,
                delta=self._delta2str[self._neighbor_deltas[index][i]]
            )
        return code

    def neighbor_state(self, neighbor_index, state_index, coord_prefix):
        """
        Generate the C code to obtain a neighbor's state by its index.

        See :meth:`Neighborhood.neighbor_coords` for details.

        """
        cell_index = self.topology.lattice.coord_to_index_code(coord_prefix)
        cell_index += " + n * " + str(state_index)
        code = "fld[{cell_index}]".format(cell_index=cell_index)
        return code


class MooreNeighborhood(OrthogonalNeighborhood):
    """
    N-dimensional Moore neighborhood implementation.

    The neighbors are all cells, sharing at least one vertex.

    """

    @OrthogonalNeighborhood.dimensions.setter
    def dimensions(self, num_dim):
        """Set the number of neighbors and their relative coordinates."""
        super_class = super(MooreNeighborhood, MooreNeighborhood)
        super_class.dimensions.fset(self, num_dim)
        self.num_neighbors = 3 ** num_dim - 1
        deltas = itertools.product([-1, 0, 1], repeat=num_dim)
        self._neighbor_deltas = [d for d in deltas if d != (0, 0)]
        self.rev_index_map = list(range(self.num_neighbors))[::-1]


class VonNeumannNeighborhood(OrthogonalNeighborhood):
    """
    N-dimensional Von Neumann neighborhood implementation.

    The neighbors are adjacent cells in all possible orthogonal directions.

    """

    @OrthogonalNeighborhood.dimensions.setter
    def dimensions(self, num_dim):
        """Set the number of neighbors and their relative coordinates."""
        super_class = super(VonNeumannNeighborhood, VonNeumannNeighborhood)
        super_class.dimensions.fset(self, num_dim)
        self.num_neighbors = 2 * num_dim
        self._neighbor_deltas = []
        for i in range(num_dim):
            delta = tuple((1 if i == j else 0 for j in range(num_dim)))
            self._neighbor_deltas.append(delta)
            delta = tuple((-1 if i == j else 0 for j in range(num_dim)))
            self._neighbor_deltas.append(delta)
            self.rev_index_map += [i * 2 + 1, i * 2]
