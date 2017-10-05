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
    supported_dimensions = [2, ]
    num_neighbors = 8

    def neighbor_coords(self, index, coord_prefix, neighbor_prefix):
        return ""

    def neighbor_state(self, neighbor_index, state_index,
                       coord_prefix, state_name):
        return ""
