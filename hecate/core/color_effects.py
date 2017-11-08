import inspect

from hecate.core.variables import Constant
from hecate.core.base import BSCA, HecateException


class ColorEffect:

    def __init__(self, func):
        self.func = func

    @property
    def _bsca(self):
        frame = inspect.currentframe()
        while frame is not None:
            for l in frame.f_locals.values():
                if hasattr(l, "__get__"):
                    continue
                if isinstance(l, BSCA):
                    return l
            frame = frame.f_back
        raise HecateException("BSCA not detected for ColorEffect")

    def __call__(self, self_var):
        self._bsca._constants.add(Constant("FADE_IN", "fade_in"))
        self._bsca._constants.add(Constant("FADE_OUT", "fade_out"))
        self._bsca._constants.add(Constant("SMOOTH_FACTOR",
                                           "smooth_factor"))

        r, g, b = self.func(self_var)
        code = """
            int new_r = %s;
            int new_g = %s;
            int new_b = %s;
            %s
            col[i] = make_int3(new_r, new_g, new_b);
        """ % (r, g, b, self.effect)
        self_var.append_code(code)


class MovingAverage(ColorEffect):

    def __call__(self, *args):
        self.effect = """
            new_r *= SMOOTH_FACTOR;
            new_g *= SMOOTH_FACTOR;
            new_b *= SMOOTH_FACTOR;
            int3 old_col = col[i];
            new_r = max(min(new_r, old_col.x + FADE_IN),
                        old_col.x - FADE_OUT);
            new_g = max(min(new_g, old_col.y + FADE_IN),
                        old_col.y - FADE_OUT);
            new_b = max(min(new_b, old_col.z + FADE_IN),
                        old_col.z - FADE_OUT);
        """
        return super(MovingAverage, self).__call__(*args)
