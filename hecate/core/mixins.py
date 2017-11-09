import inspect

import hecate.core.base


class BscaDetector:

    @property
    def _bsca(self):
        frame = inspect.currentframe()
        while frame is not None:
            for l in frame.f_locals.values():
                if hasattr(l, "__get__"):
                    continue
                if isinstance(l, hecate.core.base.BSCA):
                    return l
            frame = frame.f_back
        raise hecate.core.base.HecateException("BSCA not detected")

    @property
    def _holder_frame(self):
        # TODO: detect base class by scanning inheritance tree:
        # inspect.getclasstree(inspect.getmro(type(self)))
        frame = inspect.currentframe().f_back.f_back.f_back
        while isinstance(frame.f_locals.get('self', ''), self.base_class):
            frame = frame.f_back
        return frame

    @property
    def _holder(self):
        return self._holder_frame.f_locals['self']
