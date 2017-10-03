from hecate.core.topology.mixins import DimensionsMixin


class Border(DimensionsMixin):
    """
    Base class for all types of borders.

    """


class TorusBorder(Border):
    supported_dimensions = [2, ]
