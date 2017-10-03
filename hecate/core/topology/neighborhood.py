from hecate.core.topology.mixins import DimensionsMixin


class Neighborhood(DimensionsMixin):
    """
    Base class for all types of neighborhood.

    """


class MooreNeighborhood(Neighborhood):
    supported_dimensions = [2, ]
