"""
The collection of mixins to be used in core classes.

Would be interesting only if you are planning to hack into Xentica
core functionality.

"""
import inspect

import xentica.core.base
from xentica.core.exceptions import XenticaException

__all__ = ['BscaDetectorMixin', ]


class BscaDetectorMixin:
    """
    Add a functionlality to detect BSCA class instances holding current class.

    All methods are for private use only.

    """

    @property
    def bsca(self):
        """
        Get a BSCA instance holding current class.

        Objects tree is scanned up to top and first instance found is returned.

        """
        frame = inspect.currentframe()
        while frame is not None:
            for l in frame.f_locals.values():
                if hasattr(l, "__get__"):
                    continue
                if isinstance(l, xentica.core.base.BSCA):
                    return l
            frame = frame.f_back
        raise XenticaException("BSCA not detected")

    @property
    def _holder_frame(self):
        """
        Get a frame of class instance holding current class.

        Objects tree is scanned up to top and first instance found is returned.

        """
        # TODO: detect base class by scanning inheritance tree:
        # inspect.getclasstree(inspect.getmro(type(self)))
        frame = inspect.currentframe().f_back.f_back.f_back
        while isinstance(frame.f_locals.get('self', ''), self.base_class):
            frame = frame.f_back
        return frame

    @property
    def _holder(self):
        """Get an instance from a frame found by :meth:`_holder_frame`."""
        return self._holder_frame.f_locals['self']
