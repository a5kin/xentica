import itertools

from hecate.core.topology.mixins import DimensionsMixin


class Neighborhood(DimensionsMixin):
    """
    Base class for all types of neighborhood.

    """
    def __init__(self):
        self.num_neighbors = None
        self.topology = None
        self._delta2str = {-1: " - 1", 0: "", 1: " + 1"}

    def __len__(self):
        return self.num_neighbors


class MooreNeighborhood(Neighborhood):
    supported_dimensions = list(range(1, 100))

    @Neighborhood.dimensions.setter
    def dimensions(self, num_dim):
        super_class = super(MooreNeighborhood, MooreNeighborhood)
        super_class.dimensions.fset(self, num_dim)
        self.num_neighbors = 3 ** num_dim - 1
        deltas = itertools.product([-1, 0, 1], repeat=num_dim)
        self._neighbor_deltas = [d for d in deltas if d != (0, 0)]

    def neighbor_coords(self, index, coord_prefix, neighbor_prefix):
        code = ""
        for i in range(self.dimensions):
            code += "{neighbor}{i} = {coord}{i}{delta};\n".format(
                neighbor=neighbor_prefix, i=i,
                coord=coord_prefix,
                delta=self._delta2str[self._neighbor_deltas[index][i]]
            )
        return code

    def neighbor_state(self, neighbor_index, state_index,
                       coord_prefix, state_name):
        cell_index = self.topology.lattice.coord_to_index_code(coord_prefix)
        if state_index + 1 > 0:
            cell_index += " + n * " + str(state_index + 1)
        code = "unsigned char {state} = fld[{cell_index}];\n".format(
            state=state_name, cell_index=cell_index
        )
        return code
