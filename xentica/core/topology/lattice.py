from xentica.core.topology.mixins import DimensionsMixin

from xentica.core.mixins import BscaDetectorMixin
from xentica.core.variables import Constant


class Lattice(DimensionsMixin, BscaDetectorMixin):
    """
    Base class for all lattices.

    """
    width_prefix = "_w"

    def _define_constants_once(self):
        for i in range(self._bsca.topology.dimensions):
            constant = Constant("%s%d" % (self.width_prefix, i),
                                "size[%d]" % i)
            self._bsca.define_constant(constant)


class OrthogonalLattice(Lattice):
    supported_dimensions = list(range(1, 100))

    def index_to_coord_code(self, index_name, coord_prefix):
        self._define_constants_once()

        def wrap_format(s):
            return s.format(x=coord_prefix, i=i,
                            index=index_name, w=self.width_prefix)

        for i in range(self.dimensions):
            if i == 0:
                code = wrap_format("int _{index} = {index};\n")
                index_name = "_" + index_name
            if i < self.dimensions - 1:
                code += wrap_format("int {x}{i} = {index} % {w}{i};\n")
                code += wrap_format("{index} /= {w}{i};\n")
            else:
                code += wrap_format("int {x}{i} = {index};\n")
        return code

    def index_to_coord(self, idx, bsca):
        coord = []
        for i in range(bsca.topology.dimensions):
            if i < self.dimensions - 1:
                x_i = idx % bsca.size[i]
                idx //= bsca.size[i]
            else:
                x_i = idx
            coord.append(x_i)
        return tuple(coord)

    def coord_to_index_code(self, coord_prefix):
        self._define_constants_once()

        summands = []
        for i in range(self.dimensions):
            summand = coord_prefix + str(i)
            for j in range(i):
                summand = self.width_prefix + str(j) + " * " + summand
            summands.append(summand)
        return " + ".join(summands)

    def is_off_board_code(self, coord_prefix):
        self._define_constants_once()

        conditions = []
        for i in range(self.dimensions):
            condition = "{x}{i} < 0 || {x}{i} >= {w}{i}".format(
                x=coord_prefix, i=i, w=self.width_prefix
            )
            conditions.append(condition)
        return " || ".join(conditions)
