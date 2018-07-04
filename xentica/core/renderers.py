"""
The collection of classes implementing render logic.

The renderer takes the array of cells' colors and renders the screen
frame from it. Also, it is possible to expand a list of user actions,
adding ones specific to the renderer, like zoom, scroll etc.

The default renderer is :class:`RendererPlain`. Though there are no
other renderers yet, you may try to implement your own and apply it to
CA model as follows::

    from xentica.core import CellularAutomaton
    from xentica.core.renderers import Renderer

    class MyRenderer(Renderer):
        # ...

    class MyCA(CellularAutomaton):
        renderer = MyRenderer()
        # ...

"""
import abc

import numpy as np

from xentica.core.mixins import BscaDetectorMixin

__all__ = ['Renderer', 'RendererPlain', ]


class Renderer(BscaDetectorMixin):
    """
    Base class for all renderers.

    For correct behavior, renderer classes should be inherited from
    this class. Then at least :meth:`render_code` method should be implemented.

    However, if you are planning to add user actions specific to your
    renderer, more methods should be overridden:

    - :meth:`__init__`, where you expand a list of kernel arguments in
      ``self.args``;

    - :meth:`get_args_vals`, where you expand the list of arguments' values;

    - :meth:`setup_actions`, where you expand a dictionary of bridge actions;

    See :class:`RendererPlain` code as an example.

    """

    def __init__(self):
        """Initialize kernel arguments."""
        self.args = [
            ("int3", "*col"),
            ("int", "*img"),
        ]

    @staticmethod
    def get_args_vals(bsca):
        """
        Get a list of kernel arguments values.

        The order should correspond to ``self.args``, with the values
        themselves as either PyCUDA ``GpuArray`` or correct NumPy
        instance. Those values will be used directly as arguments to
        PyCUDA kernel execution.

        :param bsca:
            :class:`xentica.core.CellularAutomaton` instance.

        """
        args_vals = [bsca.gpu.arrays.colors, bsca.gpu.arrays.img]
        return args_vals

    @abc.abstractmethod
    def render_code(self):
        """
        Generate C code for rendering.

        At minimum, it should process cells colors stored in ``col``
        GPU-array, and store the resulting pixel's value into ``img``
        GPU-array. It can additionally use other custom arguments, if
        any set up.

        """
        return ""

    def setup_actions(self, bridge):
        """
        Expand bridge with custom user actions.

        You can do it as follows::

            class MyRenderer(Renderer):
                # ...

                @staticmethod
                def my_awesome_action():
                    def func(ca, gui):
                        # do something with ``ca`` and ``gui``
                    return func

                def setup_actions(self):
                    bridge.key_actions.update({
                        "some_key": self.my_awesome_action(),
                    })

        :param bridge:
            :class:`xentica.bridge.Bridge` instance.

        """


class RendererPlain(Renderer):
    """
    Render board as 2D plain.

    If your model has more than 2 dimensions, a projection over
    ``projection_axes`` tuple will be made. The default is two first
    axes, which corresponds to ``(0, 1)`` tuple.

    """

    def __init__(self, projection_axes=None):
        """Initialize custom kernel arguments and projection axes.

        :param projection_axes:
            A tuple with indexes of 2 axes over which a projection is
            made.  If ``None`` value is given, two first axes, ``(0, 1)``
            will be used.

        """
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
        """Extend kernel arguments values."""
        args_vals = super(RendererPlain, self).get_args_vals(bsca)
        args_vals += [
            np.int32(bsca.zoom),
            np.int32(bsca.pos[0]),
            np.int32(bsca.pos[1]),
            np.int32(bsca.width),
        ]
        return args_vals

    def setup_actions(self, bridge):
        """Extend bridge with scroll and zoom user actions."""
        bridge.key_actions.update({
            "up": self.move(0, 1),
            "down": self.move(0, -1),
            "left": self.move(-1, 0),
            "right": self.move(1, 0),
            "=": self.zoom(1),
            "-": self.zoom(-1),
        })

    @staticmethod
    def move(delta_x, delta_y):
        """
        Move over game field by some delta.

        :param dx: Delta by x-axis.
        :param dy: Delta by y-axis.

        """
        def func(model, _gui):
            """Implement move over field."""
            model.renderer.apply_move(model, delta_x, delta_y)
        return func

    @staticmethod
    def zoom(dzoom):
        """
        Zoom game field by some delta.

        :param dzoom: Delta by which field is zoomed.

        """
        def func(model, _gui):
            """Implement field zoom."""
            model.renderer.apply_zoom(model, dzoom)
        return func

    @staticmethod
    def apply_move(bsca, *args):
        """
        Apply field move action to CA class.

        :param bsca:
            :class:`xentica.core.CellularAutomaton` instance.

        """
        for i, arg in enumerate(args):
            delta = arg
            bsca.pos[i] = (bsca.pos[i] + delta) % bsca.size[i]

    @staticmethod
    def apply_zoom(bsca, dval):
        """
        Apply field zoom action to CA class.

        :param bsca:
            :class:`xentica.core.CellularAutomaton` instance.
        :param dval:
            Delta by which field is zoomed.

        """
        bsca.zoom = max(1, (bsca.zoom + dval))

    def render_code(self):
        """Implement the code for render kernel."""
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
            w=self.bsca.topology.lattice.width_prefix,
            i0=self.projection_axes[0],
            i1=self.projection_axes[1],
        )
        # sum over projected dimensions
        c_for = ""
        for i in range(self.bsca.topology.dimensions):
            if i in self.projection_axes:
                continue
            code += "num_cells_projected *= {w}{i};\n".format(
                i=i, w=self.bsca.topology.lattice.width_prefix
            )
            c_for += "for (int x{i} = 0; x{i} < {w}{i}; x{i}++) {{\n".format(
                i=i, w=self.bsca.topology.lattice.width_prefix
            )
        code += c_for
        code += """
            int ii = {coord_to_index};
            int3 c = col[ii];
            r += c.x;
            g += c.y;
            b += c.z;
        """.format(
            coord_to_index=self.bsca.topology.lattice.coord_to_index_code("x")
        )
        code += "}" * (self.bsca.topology.dimensions - 2)
        # calculate average
        by_smooth_factor = ""
        if self.bsca.is_constant("SMOOTH_FACTOR"):
            by_smooth_factor = "/ SMOOTH_FACTOR"
        code += """
            img[i * 3] = r / num_cells_projected {by_smooth_factor};
            img[i * 3 + 1] = g / num_cells_projected {by_smooth_factor};
            img[i * 3 + 2] = b / num_cells_projected {by_smooth_factor};
        """.format(by_smooth_factor=by_smooth_factor)
        return code
