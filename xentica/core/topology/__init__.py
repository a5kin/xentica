"""This package helps you build the topology for CA models.

All :class:`xentica.core.CellularAutomaton` subclasses **must** have
``Topology`` class declared inside. This class describes:

- ``dimensions``: the number of dimensions your CA model operates on.

- ``lattice``: the type of lattice of your CA board. Built-in lattice
  types are available in :mod:`xentica.core.topology.lattice` module.

- ``neighborhood``: the type of neighborhood for a single
  cell. Built-in neighborhood types are available in
  :mod:`xentica.core.topology.neighborhood` module.

- ``border``: the type of border effect, e.g. how to process off-board
  cells. Built-in border types are available in
  :mod:`xentica.core.topology.border` module.

In example, you can declare the topology for a 2-dimensional
orthogonal lattice with Moore neighborhood, wrapped to a 3-torus, as
follows::

    class Topology:

        dimensions = 2
        lattice = core.OrthogonalLattice()
        neighborhood = core.MooreNeighborhood()
        border = core.TorusBorder()

"""
