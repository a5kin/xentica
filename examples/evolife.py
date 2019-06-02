"""
`EvoLife`_ ported to Xentica.

.. _EvoLife: https://github.com/a5kin/evolife

"""
from xentica import core
from xentica import seeds
from xentica.tools import xmath
from xentica.core import color_effects
from xentica.tools.color import GenomeColor
from xentica.tools.genetics import genome_crossover
from xentica.tools.rules import LifeLike
from xentica.seeds.random import RandInt

from examples.base import RegularCA, RegularExperiment
from examples.base import run_simulation


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
    energy = core.IntegerProperty(max_val=255 * 2)
    rule = core.TotalisticRuleProperty(outer=True)
    rng = core.RandomProperty()
    death_speed = core.Parameter(default=15)
    max_genes = core.Parameter(default=9)
    mutation_prob = core.Parameter(default=.0)

    def __init__(self, *args, legacy_coloring=True):
        """Support legacy coloring as needed."""
        self._legacy_coloring = legacy_coloring
        super().__init__(*args)

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
            nbr_energy = self.neighbors[i].buffer.energy
            nbr_rule = self.neighbors[i].buffer.rule
            num_neighbors += xmath.min(1, (nbr_energy + nbr_rule))
        is_sustained = core.IntegerVariable()
        is_sustained += self.main.rule.is_sustained(num_neighbors)

        # test if cell is born
        fitnesses = []
        for i in range(len(self.buffers)):
            fitnesses.append(core.IntegerVariable(name="fit%d" % i))
        num_parents = core.IntegerVariable()
        for gene in range(len(self.buffers)):
            num_parents *= 0  # hack for re-init variable
            for i in range(len(self.buffers)):
                nbr_energy = self.neighbors[i].buffer.energy
                nbr_rule = self.neighbors[i].buffer.rule
                is_alive = xmath.min(1, (nbr_energy + nbr_rule))
                is_fit = self.neighbors[i].buffer.rule.is_born(gene + 1)
                num_parents += is_alive * is_fit
            fitnesses[gene] += num_parents * (num_parents == (gene + 1))
        num_fit = core.IntegerVariable()
        num_fit += xmath.max(*fitnesses)

        # neighbor's genomes crossover
        genomes = []
        for i in range(len(self.buffers)):
            genomes.append(core.IntegerVariable(name="genome%d" % i))
        for i in range(len(self.buffers)):
            is_fit = self.neighbors[i].buffer.rule.is_born(num_fit)
            genomes[i] += self.neighbors[i].buffer.rule * is_fit
        num_genes = self.main.rule.bit_width
        old_rule = core.IntegerVariable()
        old_rule += self.main.rule
        old_energy = core.IntegerVariable()
        old_energy += self.main.energy
        new_genome = genome_crossover(
            self.main, num_genes, *genomes,
            max_genes=self.meta.max_genes,
            mutation_prob=self.meta.mutation_prob
        )
        self.main.rule = new_genome + self.main.rule * (new_genome == 0)

        # new energy value
        self.main.energy *= 0
        is_live = core.IntegerVariable()
        old_live = (old_energy + old_rule) == 0
        is_live += (old_energy < 0xff) & (old_live | is_sustained)
        old_dead = (old_energy + old_rule) != 0
        new_energy = old_energy + self.meta.death_speed * old_dead
        self.main.energy = new_energy * (self.main.rule == old_rule) + \
            self.main.energy * (self.main.rule != old_rule)
        self.main.rule = old_rule * (self.main.rule == old_rule) + \
            self.main.rule * (self.main.rule != old_rule)
        self.main.energy *= is_live
        self.main.rule *= is_live

    @color_effects.MovingAverage
    def color(self):
        """Render cell's genome as hue/sat, cell's energy as value."""
        if self._legacy_coloring:
            red, green, blue = GenomeColor.modular(self.main.rule >> 1, 360)
        else:
            red, green, blue = GenomeColor.positional(self.main.rule,
                                                      self.main.rule.bit_width)
        is_live = (self.main.rule > 0) * (self.main.energy < 255)
        energy = (255 - self.main.energy) * is_live
        red = xmath.int(red * energy)
        green = xmath.int(green * energy)
        blue = xmath.int(blue * energy)
        return (red, green, blue, )


class CrossbreedingExperiment(RegularExperiment):
    """Classic experiment for legacy EvoLife, where 'Bliambas' may form."""

    word = "FUTURE BREEDING MACHINE 2041"
    size = (1920, 1080)
    zoom = 1
    fade_in = 18
    fade_out = 3
    seed_diamoeba = seeds.patterns.BigBang(
        pos=(380, 280),
        size=(440, 340),
        vals={
            "energy": 0,
            "rule": LifeLike.golly2int("B35678/S5678") * RandInt(0, 1),
            "rng": RandInt(0, 2 ** 16 - 1)
        }
    )
    seed_conway = seeds.patterns.BigBang(
        pos=(0, 0),
        size=(1280, 720),
        vals={
            "energy": 0,
            # you may use unary operators
            "rule": +RandInt(0, 1) * LifeLike.golly2int("B3/S23"),
            # as well as reflected expressions
            "rng": 0 + RandInt(0, 2 ** 16 - 1)
        }
    )
    seed_rng = seeds.patterns.PrimordialSoup(
        vals={
            "rng": RandInt(0, 2 ** 16 - 1)
        }
    )
    # chain ordering matters, since areas are rewriting each other in order
    seed = seed_rng + seed_conway + seed_diamoeba


class CrossbreedingExperiment2(CrossbreedingExperiment):
    """Same crossbreeding experiment with different meta-parameters."""
    death_speed = 25
    max_genes = 9
    mutation_prob = 0.0001


class BigBangExperiment(RegularExperiment):
    """Experiment, with smaller field, for quicker benchmark."""
    death_speed = 0
    max_genes = 9
    seed_bang = seeds.patterns.BigBang(
        pos=(0, 0),
        size=(100, 100),
        vals={
            "energy": RandInt(0, 255),
            "rule": RandInt(0, 2 ** 18 - 1) & 0b111111110111111110,
            "rng": RandInt(0, 2 ** 16 - 1)
        }
    )
    seed_rng = seeds.patterns.PrimordialSoup(
        vals={
            "rng": RandInt(0, 2 ** 16 - 1)
        }
    )
    seed = seed_rng + seed_bang


if __name__ == "__main__":
    run_simulation(EvoLife, CrossbreedingExperiment2)
