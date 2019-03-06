"""
`EvoLife`_ ported to Xentica.

.. _EvoLife: https://github.com/a5kin/evolife

"""
from xentica import core
from xentica import seeds
from xentica.tools.xmath import xmin, xmax
from xentica.core import color_effects
from xentica.seeds.random import RandInt

from .base import RegularCA, RegularExperiment
from .base import run_simulation


class EvoLife(RegularCA):
    """
    Life-like cellular automaton with evolutionary rules for each cell.

    Rules are:
    - Each living cell has its own birth/sustain ruleset and an energy
      level;
    - Cell is loosing all energy if number of neighbours is not in its
      sustain rule;
    - Cell is born with max energy if there are exactly N neighbours
      with N in their birth rule;
        - Same is applied for living cells (re-occupation case), if
          new genome is different;
    - If there are several birth situations with different N possible,
      we choose one with larger N;
    - Newly born cell's ruleset calculated as crossover between
      'parent' cells rulesets;
    - If cell is involved in breeding as a 'parent', it's loosing
      BIRTH_COST units of energy per each non-zero gene passed;
        - This doesn't apply in re-occupation case;
    - Every turn, cell is loosing DEATH_SPEED units of energy;
    - Cell with zero energy is dying;
    - Cell cannot have more than MAX_GENES non-zero genes in ruleset.

    """
    energy = core.IntegerProperty(max_val=255)
    rule = core.LifelikeRuleProperty()
    rng = core.RandomProperty()

    def emit(self):
        """Broadcast the state to all neighbors."""
        for i in range(len(self.buffers)):
            self.buffers[i].energy = self.main.energy
            self.buffers[i].rule = self.main.rule

    def absorb(self):
        """Apply EvoLife dynamics."""
        # test if cell is sustained
        num_neighbors = core.IntegerVariable()
        for i in range(len(self.buffers)):
            num_neighbors += xmin(1, self.neighbors[i].buffer.energy)
        is_sustained = self.main.rule.is_sustained(num_neighbors)

        # test if cell is born
        fitnesses = [core.IntegerVariable() for _ in range(len(self.buffers))]
        num_parents = core.IntegerVariable()
        for gene in range(len(self.buffers)):
            num_parents *= 0  # hack for re-init variable
            for i in range(len(self.buffers)):
                is_alive = xmin(1, self.neighbors[i].buffer.energy)
                is_fit = self.neighbors[i].buffer.rule.is_born(gene)
                num_parents += is_alive * is_fit
            fitnesses[gene] = num_parents * (num_parents == gene)
        num_fit = xmax(*fitnesses)

        # new energy value
        self.main.energy = xmax(0, self.main.energy - 1)
        self.main.energy *= is_sustained
        self.main.energy |= 255 * (num_fit > 0)

        # TODO: genomes crossover

    @color_effects.MovingAverage
    def color(self):
        """Render cell's energy in monochrome."""
        # TODO: render cell in HSV
        red = self.main.energy
        green = self.main.energy
        blue = self.main.energy
        return (red, green, blue)


class BigBangExperiment(RegularExperiment):
    """Default experiment for legacy EvoLife."""

    word = "BANG! BANG! BANG!"
    seed = seeds.patterns.BigBang(
        pos=(320, 180),
        size=(100, 100),
        vals={
            "energy": RandInt(0, 1) * RandInt(1, 255),
            "rule": RandInt(0, 2 ** 17 - 1),
            "rng": RandInt(0, 2 ** 16 - 1)
        }
    )


if __name__ == "__main__":
    run_simulation(EvoLife, BigBangExperiment)
