import numpy as np

from xentica.core.mixins import BscaDetectorMixin


class Renderer(BscaDetectorMixin):
    """
    Base class for renderers.

    """
    def __init__(self):
        self.args = [
            ("int3", "*col"),
            ("int", "*img"),
        ]

    def get_args_vals(self, bsca):
        args_vals = [bsca.colors_gpu, bsca.img_gpu]
        return args_vals

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

    def get_args_vals(self, bsca):
        args_vals = super(RendererPlain, self).get_args_vals(bsca)
        args_vals += [
            np.int32(bsca.zoom),
            np.int32(bsca.pos[0]),
            np.int32(bsca.pos[1]),
            np.int32(bsca.width),
        ]
        return args_vals

    def setup_actions(self, bridge):
        bridge.key_actions.update({
            "up": self.move(0, 1),
            "down": self.move(0, -1),
            "left": self.move(-1, 0),
            "right": self.move(1, 0),
            "=": self.zoom(1),
            "-": self.zoom(-1),
        })

    @staticmethod
    def move(dx, dy):
        def func(ca, gui):
            ca.renderer.apply_move(ca, dx, dy)
        return func

    @staticmethod
    def zoom(dzoom):
        def func(ca, gui):
            ca.renderer.apply_zoom(ca, dzoom)
        return func

    @staticmethod
    def apply_move(bsca, *args):
        for i in range(len(args)):
            delta = args[i]
            bsca.pos[i] = (bsca.pos[i] + delta) % bsca.size[i]

    @staticmethod
    def apply_zoom(bsca, dval):
        bsca.zoom = max(1, (bsca.zoom + dval))

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
            int num_cells_projected = 1;
        """.format(
            x="x%d" % self.projection_axes[0],
            y="x%d" % self.projection_axes[1],
            w=self._bsca.topology.lattice.width_prefix,
            i0=self.projection_axes[0],
            i1=self.projection_axes[1],
        )
        # sum over projected dimensions
        c_for = ""
        for i in range(self._bsca.topology.dimensions):
            if i in self.projection_axes:
                continue
            code += "num_cells_projected *= {w}{i};\n".format(
                i=i, w=self._bsca.topology.lattice.width_prefix
            )
            c_for += "for (int x{i} = 0; x{i} < {w}{i}; x{i}++) {{\n".format(
                i=i, w=self._bsca.topology.lattice.width_prefix
            )
        code += c_for
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
        by_smooth_factor = ""
        if self._bsca.is_constant("SMOOTH_FACTOR"):
            by_smooth_factor = "/ SMOOTH_FACTOR"
        code += """
            img[i * 3] = r / num_cells_projected {by_smooth_factor};
            img[i * 3 + 1] = g / num_cells_projected {by_smooth_factor};
            img[i * 3 + 2] = b / num_cells_projected {by_smooth_factor};
        """.format(by_smooth_factor=by_smooth_factor)
        return code
