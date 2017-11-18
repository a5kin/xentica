from hecate.core.mixins import BscaDetectorMixin


class Renderer(BscaDetectorMixin):
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
        # calculate projection plain coordinates
        code = """
            int {x} = (int) (((float) (i % width)) / (float) zoom) + dx;
            int {y} = (int) (((float) (i / width)) / (float) zoom) + dy;
            if ({x} < 0) {x} = {w}{i0} - (-{x} % {w}{i0});
            if ({x} >= {w}{i0}) {x} = {x} % {w}{i0};
            if ({y} < 0) {y} = {w}{i1} - (-{y} % {w}{i1});
            if ({y} >= {w}{i1}) {y} = {y} % {w}{i1};
            float r = 0, g = 0, b = 0;
        """.format(
            x="x%d" % self.projection_axes[0],
            y="x%d" % self.projection_axes[1],
            w=self._bsca.topology.lattice.width_prefix,
            i0=self.projection_axes[0],
            i1=self.projection_axes[1],
        )
        # sum over projected dimensions
        num_cells_projected = 1
        for i in range(self._bsca.topology.dimensions):
            if i in self.projection_axes:
                continue
            num_cells_projected *= self._bsca.size[i]
            code += "for (int x{i} = 0; x{i} < {w}{i}; x{i}++) {\n".format(
                i=i, w=self._bsca.topology.lattice.width_prefix
            )
        code += """
            int ii = {coord_to_index};
            int3 c = col[ii];
            r += c.x;
            g += c.y;
            b += c.z;
        """.format(
            coord_to_index=self._bsca.topology.lattice.coord_to_index_code("x")
        )
        code += "}" * (self._bsca.topology.dimensions - 2)
        # calculate average
        code += """
            img[i * 3] = r / {num} / SMOOTH_FACTOR;
            img[i * 3 + 1] = g / {num} / SMOOTH_FACTOR;
            img[i * 3 + 2] = b / {num} / SMOOTH_FACTOR;
        """.format(num=num_cells_projected)
        return code
