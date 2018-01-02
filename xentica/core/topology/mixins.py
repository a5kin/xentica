"""
The collection of mixins to be used in core classes.

Would be interesting only if you are planning to hack into Xentica
core functionality.

"""
from xentica.core.exceptions import XenticaException


class DimensionsMixin:
    """
    The base functionality for classes, operating on a number of dimensions.

    Adds ``dimensions`` property to a class, and checks it
    automatically over a list of allowed dimensions.

    """

    #: A list of integers, containing supported dimensionality.
    #: You must set it manually for every class using :class:`DimensionsMixin`.
    supported_dimensions = []

    def __init__(self):
        """Initialize private variables."""
        self._dimensions = None

    def allowed_dimension(self, num_dim):
        """
        Test if particular dimensionality is allowed.

        :param num_dim:
            Numbers of dimensions to test

        :returns:
            Boolean value, either dimensionality is allowed or not.

        """
        return num_dim in self.supported_dimensions

    @property
    def dimensions(self):
        """Get a number of dimensions."""
        return self._dimensions

    @dimensions.setter
    def dimensions(self, num_dim):
        """Set a number of dimensions and check if it is allowed."""
        if not self.allowed_dimension(num_dim):
            msg = "%d-D %s is not supported."
            msg = msg % (num_dim, self.__class__.__name__)
            raise XenticaException(msg)
        self._dimensions = num_dim
