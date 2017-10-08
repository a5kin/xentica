from hecate.core.topology.mixins import DimensionsMixin


class Lattice(DimensionsMixin):
    """
    Base class for all lattices.

    """
    width_prefix = "_w"


class OrthogonalLattice(Lattice):
    supported_dimensions = list(range(1, 100))

    def index_to_coord_code(self, index_name, coord_prefix):

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

    def coord_to_index_code(self, coord_prefix):
        summands = []
        for i in range(self.dimensions):
            summand = coord_prefix + str(i)
            for j in range(i):
                summand = self.width_prefix + str(j) + " * " + summand
            summands.append(summand)
        return " + ".join(summands)
