from hecate.core.topology.mixins import DimensionsMixin


class Lattice(DimensionsMixin):
    """
    Base class for all lattices.

    """


class OrthogonalLattice(Lattice):
    supported_dimensions = [2, ]
