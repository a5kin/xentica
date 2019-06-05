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
   - ``core.TotalisticRuleProperty`` →
     :class:`xentica.core.properties.TotalisticRuleProperty`
   - ``core.RandomProperty`` →
     :class:`xentica.core.properties.RandomProperty`

- **Parameters**
   - ``core.Parameter`` →
     :class:`xentica.core.parameters.Parameter`

- **Variables**
   - ``core.IntegerVariable`` →
     :class:`xentica.core.variables.IntegerVariable`
   - ``core.FloatVariable`` →
     :class:`xentica.core.variables.FloatVariable`

The classes listed above are all you need to build CA models and
experiments with Xentica, unless you are planning to implement custom
core features like new lattices, borders, etc.

"""
from xentica.core.base import CellularAutomaton
from xentica.core.properties import (
    IntegerProperty,
    TotalisticRuleProperty,
    RandomProperty,
)
from xentica.core.variables import (
    IntegerVariable, FloatVariable,
)
from xentica.core.parameters import (
    Parameter,
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
    'TotalisticRuleProperty',
    'RandomProperty',
    'Parameter',
    'IntegerVariable',
    'FloatVariable',
    'OrthogonalLattice',
    'MooreNeighborhood',
    'VonNeumannNeighborhood',
    'TorusBorder',
    'StaticBorder',
    'Experiment',
]
