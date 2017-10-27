def color_effect(func, effect):
    def wrapper(self_var):
        r, g, b = func(self_var)
        code = """
            int new_r = %s;
            int new_g = %s;
            int new_b = %s;
            %s
            col[i] = make_int3(new_r, new_g, new_b);
        """ % (r, g, b, effect)
        return code
    return wrapper


def moving_average(func):
    effect = """
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
    return color_effect(func, effect)
