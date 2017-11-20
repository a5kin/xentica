from xentica.core.variables import Constant
from xentica.core.mixins import BscaDetectorMixin


class ColorEffect(BscaDetectorMixin):

    def __init__(self, func):
        self.func = func

    def __call__(self, self_var):
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
        self._bsca.define_constant(Constant("FADE_IN", "fade_in"))
        self._bsca.define_constant(Constant("FADE_OUT", "fade_out"))
        self._bsca.define_constant(Constant("SMOOTH_FACTOR",
                                            "smooth_factor"))
        self._bsca.fade_in = 255
        self._bsca.fade_out = 255
        self._bsca.smooth_factor = 1
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
