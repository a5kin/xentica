"""
Conserved Life implemented with Xentica.

"""
import random
import importlib

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


class ConservedLife(RegularCA):
    """
    Energy conserved model, acting on Life-like rules.

    Main rules are:
    - Each cell has an integer energy level;
    - Cell is 'full' when its energy level is greater or
      equal to some treshold value, or it's 'empty' otherwise;
    - Full cell must spread a part of its energy, that is over
      the treshold value, or when it's 'dying';
    - Empty cell must spread a part of its energy if it's not
      'birthing';
    - Cell is 'dying' if number of 'full' neighbors is not in
      'sustain' ruleset;
    - Cell is 'birthing' if number of 'full' neighbors is in
      'birth' ruleset;
    - When spreading an energy, cell firstly chooses 'birthing'
      neighbors as targets, then 'empty' neighbors;
    - If there are several equal targets to spread an energy, cell is
      choosing the one in 'stochastic' (PRNG) way, keeping the whole
      automaton deterministic.

    Additional rules for evolutionary setting:
    - Each cell has a Life-like rule encoded into its genome;
    - If cell is 'empty', its genome is calculated as a crossover
      between all neighbors, that passed an energy to it;
    - Each cell is operating over its own rule (genome), when
      calculating dying/birthing status.

    """
    energy = core.IntegerProperty(max_val=2 ** 14 - 1)
    new_energy = core.IntegerProperty(max_val=2 ** 14 - 1)
    birthing = core.IntegerProperty(max_val=1)
    rule = core.TotalisticRuleProperty(outer=True)
    rng = core.RandomProperty()
    death_speed = core.Parameter(default=1)
    full_treshold = core.Parameter(default=1)
    max_genes = core.Parameter(default=9)
    mutation_prob = core.Parameter(default=.0)

    def __init__(self, *args, legacy_coloring=True):
        """Support legacy coloring as needed."""
        self._legacy_coloring = legacy_coloring
        super().__init__(*args)

    def emit(self):
        """Apply ConcervedLife dynamics."""
        self.main.new_energy = 1 * self.main.energy

        # calculate number of full neighbors
        num_full = core.IntegerVariable()
        for i in range(len(self.buffers)):
            is_full = self.neighbors[i].main.energy >= self.meta.full_treshold
            num_full += is_full

        # decide, do cell need to spread an energy
        me_full = self.main.energy >= self.meta.full_treshold
        me_empty = self.main.energy < self.meta.full_treshold
        me_dying = xmath.int(self.main.rule.is_sustained(num_full)) == 0
        me_dying = xmath.int(me_dying)
        me_birthing = self.main.rule.is_born(num_full)
        has_free_energy = self.main.energy > self.meta.full_treshold
        has_free_energy = xmath.int(has_free_energy)
        need_spread_full = 1 * me_full * (has_free_energy | me_dying)
        need_spread_empty = 1 * me_empty * (xmath.int(me_birthing) == 0)
        need_spread = need_spread_full + need_spread_empty

        # search the direction to spread energy
        denergy = core.IntegerVariable()
        energy_passed = core.IntegerVariable()
        energy_passed *= 1
        gate = core.IntegerVariable()
        gate += xmath.int(self.main.rng.uniform * 8)
        gate_final = core.IntegerVariable()
        treshold = self.meta.full_treshold
        for i in range(len(self.buffers) * 3):
            i_valid = (i >= gate) * (i < gate + len(self.buffers) * 2)
            is_birthing = self.neighbors[i % 8].main.birthing * (i < 8 + gate)
            is_empty = self.neighbors[i % 8].main.energy < treshold
            is_empty = is_empty * (i >= 8 + gate)
            is_fit = is_empty + is_birthing
            denergy *= 0
            denergy += xmath.min(self.main.new_energy, self.meta.death_speed)
            denergy *= need_spread * is_fit * (energy_passed == 0)
            denergy *= i_valid
            gate_final *= (energy_passed != 0)
            gate_final += (i % 8) * (energy_passed == 0)
            energy_passed += denergy

        # spread the energy in chosen direction
        for i in range(len(self.buffers)):
            gate_fit = i == gate_final
            self.buffers[i].energy = energy_passed * gate_fit
            self.buffers[i].rule = self.main.rule * gate_fit

        self.main.new_energy -= energy_passed

    def absorb(self):
        """Absorb an energy from neighbors."""
        # absorb incoming energy and calculate 'birthing' status
        incoming_energy = core.IntegerVariable()
        num_full = core.IntegerVariable()
        treshold = self.meta.full_treshold
        for i in range(len(self.buffers)):
            incoming_energy += self.neighbors[i].buffer.energy
            is_full = self.neighbors[i].main.new_energy >= treshold
            num_full += is_full

        self.main.energy = self.main.new_energy + incoming_energy
        self.main.birthing = self.main.rule.is_born(num_full)
        self.main.birthing *= self.main.energy < self.meta.full_treshold

        # neighbor's genomes crossover
        genomes = []
        for i in range(len(self.buffers)):
            genomes.append(core.IntegerVariable(name="genome%d" % i))
        for i in range(len(self.buffers)):
            is_fit = self.neighbors[i].buffer.energy > 0
            is_fit = is_fit * (self.neighbors[i].buffer.rule > 0)
            genomes[i] += self.neighbors[i].buffer.rule * is_fit
        num_genes = self.main.rule.bit_width
        new_genome = genome_crossover(
            self.main, num_genes, *genomes,
            max_genes=self.meta.max_genes,
            mutation_prob=self.meta.mutation_prob
        ) * (self.main.energy < self.meta.full_treshold)
        self.main.rule = new_genome + self.main.rule * (new_genome == 0)

    @color_effects.MovingAverage
    def color(self):
        """Render cell's genome as hue/sat, cell's energy as value."""
        if self._legacy_coloring:
            red, green, blue = GenomeColor.modular(self.main.rule >> 1, 360)
        else:
            red, green, blue = GenomeColor.positional(self.main.rule,
                                                      self.main.rule.bit_width)
        shade = xmath.min(self.main.energy, self.meta.full_treshold)
        shade = shade * 255 / self.meta.full_treshold
        red = xmath.int(red * shade)
        green = xmath.int(green * shade)
        blue = xmath.int(blue * shade)
        return (red, green, blue, )


class ConservedLife4D(ConservedLife):
    """ConservedLife variant in 4D"""

    class Topology(ConservedLife.Topology):
        """Von Neumann topology in 4D."""
        dimensions = 4
        neighborhood = core.VonNeumannNeighborhood()


class BigBangExperiment(RegularExperiment):
    """Experiment, with smaller field, for quicker benchmark."""
    death_speed = 1
    full_treshold = 16
    max_genes = 13
    fade_in = 255
    fade_out = 255
    smooth_ratio = 1
    rule_mask = LifeLike.golly2int("B12345678/S12345678")
    seed = seeds.patterns.PrimordialSoup(
        vals={
            "energy": 0,
            "new_energy": 0,
            "rule": random.randint(0, 2 ** 18 - 1) & rule_mask,
            "rng": RandInt(0, 2 ** 16 - 1),
        }
    )
    for i in range(6 * 3):
        seed += seeds.patterns.BigBang(
            pos=(30 + (i % 6) * 100, 40 + (i // 6) * 100),
            size=(80, 80),
            vals={
                "energy": 16 * RandInt(0, 1),
                "new_energy": 16 * RandInt(0, 1),
                "rule": random.randint(0, 2 ** 18 - 1) & rule_mask,
                "rng": RandInt(0, 2 ** 16 - 1),
            }
        )


class BigBangExperiment4D(RegularExperiment):
    """Experiment in 4D space."""
    size = (640, 360, 3, 3)
    death_speed = 1
    full_treshold = 16
    max_genes = 5
    fade_in = 1
    fade_out = 1
    smooth_ratio = 10
    rule_back = LifeLike.golly2int("B12/S124")
    seed = seeds.patterns.PrimordialSoup(
        vals={
            "energy": 2,
            "new_energy": 2,
            "rule": rule_back,
            "rng": RandInt(0, 2 ** 16 - 1),
        }
    )
    for i in range(6 * 3):
        add_rule = str(random.randint(4, 4)).replace("2", "")
        rule_area = LifeLike.golly2int("B12/S12" + add_rule)
        seed += seeds.patterns.BigBang(
            pos=(30 + (i % 6) * 100, 40 + (i // 6) * 100, 0, 0),
            size=(80, 80, 1, 1),
            vals={
                "energy": 1 * 273 * RandInt(0, 1),
                "new_energy": 1 * 273 * RandInt(0, 1),
                "rule": rule_area,
                "rng": RandInt(0, 2 ** 16 - 1),
            }
        )


def test_energy_conservation():
    """Test that amount of energy is conserved inside a system."""
    numpy = importlib.import_module("numpy")
    model = ConservedLife4D(BigBangExperiment4D)
    total_energy_prev = -1
    for _ in range(50000):
        model.step()
        cells = model.gpu.arrays.cells.get()[:model.cells_num]
        total_energy = int(numpy.sum(cells & (2 ** 14 - 1)))
        max_energy = int(numpy.max(cells & (2 ** 14 - 1)))
        if total_energy_prev >= 0 and total_energy != total_energy_prev:
            print(
                "Timestep %d, %d != %d" % (
                    model.timestep,
                    total_energy_prev,
                    total_energy,
                )
            )
        print("Max energy:", max_energy)
        total_energy_prev = total_energy


def render():
    """Render a video."""
    mp_editor = importlib.import_module("moviepy.editor")

    def make_frame(_):
        num_iters = 34
        for _ in range(num_iters):
            make_frame.model.step()
        width, height = make_frame.model.width, make_frame.model.height
        frame = make_frame.model.render().reshape((height, width, 3))
        return frame

    width, height = BigBangExperiment.size
    zoom = BigBangExperiment.zoom
    width, height = width * zoom, height * zoom
    make_frame.model = ConservedLife4D(BigBangExperiment4D)
    make_frame.model.set_viewport((width, height))

    animation = mp_editor.VideoClip(make_frame, duration=80)
    animation.write_videofile("conserved_life04-2.mp4", fps=24)


if __name__ == "__main__":
    # run_simulation(ConservedLife, BigBangExperiment)
    run_simulation(ConservedLife4D, BigBangExperiment4D)
    # test_energy_conservation()
    # render()
