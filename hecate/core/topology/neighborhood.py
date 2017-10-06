from hecate.core.topology.mixins import DimensionsMixin


class Neighborhood(DimensionsMixin):
    """
    Base class for all types of neighborhood.

    """
    num_neighbors = 0

    def set_border(self, border_object):
        self.border = border_object

    def __len__(self):
        return self.num_neighbors


class MooreNeighborhood(Neighborhood):
    supported_dimensions = list(range(1, 100))
    num_neighbors = None

    @Neighborhood.dimensions.setter
    def dimensions(self, num_dim):
        super_class = super(MooreNeighborhood, MooreNeighborhood)
        super_class.dimensions.fset(self, num_dim)
        self.num_neighbors = 3 ** num_dim - 1

    def neighbor_coords(self, index, coord_prefix, neighbor_prefix):
        return ""

    def neighbor_state(self, neighbor_index, state_index,
                       coord_prefix, state_name):
        return ""
