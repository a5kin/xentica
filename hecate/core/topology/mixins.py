from hecate.core.base import HecateException


class DimensionsMixin:

    supported_dimensions = []

    def allowed_dimension(self, num_dim):
        return num_dim in self.supported_dimensions

    def set_dimensions(self, num_dim):
        if not self.allowed_dimension(num_dim):
            msg = "%d-D %s is not supported."
            msg = msg % (num_dim, self.__class__.__name__)
            raise HecateException(msg)
