# -*- coding: utf-8 -*-
"""
Xentica core functionality is available via modules from this package.

In addition, you may use ``core`` package as a shortcut to the main
classes of the framework.

- **Base classes**
   - ``core.CellularAutomaton`` →
     :class:`xentica.core.base.CellularAutomaton`
   - ``core.Experiment`` →
     :class:`xentica.core.experiment.Experiment`

- **Lattices**
   - ``core.OrthogonalLattice`` →
     :class:`xentica.core.topology.lattice.OrthogonalLattice`

- **Neighborhoods**
   - ``core.MooreNeighborhood`` →
     :class:`xentica.core.topology.neighborhood.MooreNeighborhood`
   - ``core.VonNeumannNeighborhood`` →
     :class:`xentica.core.topology.neighborhood.VonNeumannNeighborhood`

- **Borders**
   - ``core.TorusBorder`` →
     :class:`xentica.core.topology.border.TorusBorder`
   - ``core.StaticBorder`` →
     :class:`xentica.core.topology.border.StaticBorder`

- **Properties**
   - ``core.IntegerProperty`` →
     :class:`xentica.core.properties.IntegerProperty`

- **Variables**
   - ``core.IntegerVariable`` →
     :class:`xentica.core.variables.IntegerVariable`

The classes listed above are all you need to build CA models and
experiments with Xentica, unless you are planning to implement custom
core features like new lattices, borders, etc.

"""
from xentica.core.base import CellularAutomaton
from xentica.core.properties import (
    IntegerProperty,
)
from xentica.core.variables import (
    IntegerVariable,
)
from xentica.core.topology.lattice import (
    OrthogonalLattice,
)
from xentica.core.topology.neighborhood import (
    MooreNeighborhood, VonNeumannNeighborhood
)
from xentica.core.topology.border import (
    TorusBorder, StaticBorder,
)
from xentica.core.experiment import Experiment

__all__ = [
    'CellularAutomaton',
    'IntegerProperty',
    'IntegerVariable',
    'OrthogonalLattice',
    'MooreNeighborhood',
    'VonNeumannNeighborhood',
    'TorusBorder',
    'StaticBorder',
    'Experiment',
]
