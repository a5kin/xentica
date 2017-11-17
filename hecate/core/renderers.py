class Renderer:
    """
    Base class for renderers.

    """
    def __init__(self):
        self.args = [
            ("int3", "*col"),
            ("int", "*img"),
        ]

    def render_code(self):
        return ""


class RendererPlain(Renderer):
    """
    Render board as 2D plain, and make projection
    from higher dimensions as needed.

    """
    def __init__(self, projection_axes=None):
        super(RendererPlain, self).__init__()
        self.args += [
            ("int", "zoom"),
            ("int", "dx"),
            ("int", "dy"),
            ("int", "width"),
        ]
        self.projection_axes = projection_axes
        if self.projection_axes is None:
            self.projection_axes = (0, 1)

    def render_code(self):
        # hardcoded pure 2D case
        code = """
            int x = (int) (((float) (i % width)) / (float) zoom) + dx;
            int y = (int) (((float) (i / width)) / (float) zoom) + dy;
            if (x < 0) x = _w0 - (-x % _w0);
            if (x >= _w0) x = x % _w0;
            if (y < 0) y = _w1 - (-y % _w1);
            if (y >= _w1) y = y % _w1;
            int ii = x + y * _w0;

            int3 c = col[ii];
            img[i * 3] = c.x / SMOOTH_FACTOR;
            img[i * 3 + 1] = c.y / SMOOTH_FACTOR;
            img[i * 3 + 2] = c.z / SMOOTH_FACTOR;
        """
        return code
