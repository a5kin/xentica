from xentica.core.base import XenticaException


class DimensionsMixin:

    supported_dimensions = []

    def __init__(self):
        self._dimensions = None

    def allowed_dimension(self, num_dim):
        return num_dim in self.supported_dimensions

    @property
    def dimensions(self):
        return self._dimensions

    @dimensions.setter
    def dimensions(self, num_dim):
        if not self.allowed_dimension(num_dim):
            msg = "%d-D %s is not supported."
            msg = msg % (num_dim, self.__class__.__name__)
            raise XenticaException(msg)
        self._dimensions = num_dim
