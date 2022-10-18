"""
Rough experiment with per cell poisson process.

"""
import importlib

from xentica import core
from xentica import seeds
from xentica.core import color_effects
from xentica.tools import xmath


class PoissonWalk(core.CellularAutomaton):
    """
    CA with Poisson process per each cell.

    More energy cell has, more likely it will transfer energy
    to the neighbour cell.

    """

    energy = core.IntegerProperty(max_val=2 ** 16 - 1)
    gate = core.IntegerProperty(max_val=2 ** 3 - 1)
    interval = core.IntegerProperty(max_val=2 ** 12 - 1)
    rng = core.RandomProperty()
    default_denergy = core.Parameter(default=1)

    class Topology:
        """Standard Moore neighbourhood with torus topology."""
        dimensions = 2
        lattice = core.OrthogonalLattice()
        neighborhood = core.MooreNeighborhood()
        border = core.TorusBorder()

    def emit(self):
        """
        Implement the logic of emit phase.

        Each cell is a Poisson process that fires an energy in the
        gate direction, when Poisson event occurs.

        The rate of events depends on cell's energy level: more
        energy, higher the rate.

        The amount of energy depends on the cell's own "valency" and
        its gated neighbour's "valency" (energy % 8).

        When neighbour's valency is 0, cell spreads the amount of
        energy, equal to its own valency. Otherwise, it spreads some
        default amount of energy.

        Cells are also spreading their gate values when event occurs,
        so they are "syncing" with neighbours.

        """
        shade = xmath.min(255, self.main.energy)
        lmb = xmath.float(self.main.interval) / xmath.float(256 - shade)
        prob = 1 - xmath.exp(-lmb)
        fired = core.IntegerVariable()
        fired += xmath.int(self.main.rng.uniform < prob)
        fired *= self.main.energy > 0
        denergy = xmath.min(self.default_denergy, shade)
        gate = self.main.gate

        mutated = core.IntegerVariable()
        mutated += xmath.int(self.main.rng.uniform < 0.0001)
        denergy = denergy - 1 * (denergy > 0) * mutated
        energy_passed = core.IntegerVariable()
        for i, buf in enumerate(self.buffers):
            valency1 = self.neighbors[i].main.energy % 8
            valency2 = self.main.energy % 8
            gate_fit = (i == gate)
            full_transition = denergy * (valency1 != 0 | valency2 == 0)
            potent_transition = valency2 * (valency1 == 0)
            denergy_fin = full_transition + potent_transition
            energy_passed += denergy_fin * fired * gate_fit
            buf.energy = energy_passed * fired * gate_fit
            buf.gate = (self.main.gate + 7) * fired * gate_fit

        self.main.interval = (self.main.interval + 1) * (energy_passed == 0)
        self.main.energy -= energy_passed * (energy_passed > 0)
        self.main.gate = (self.main.gate + 1) % 8

    def absorb(self):
        """
        Implement the logic of absorb phase.

        Each cell just sums incoming energy and gate values.

        """
        incoming_energy = core.IntegerVariable()
        incoming_gate = core.IntegerVariable()
        for i in range(len(self.buffers)):
            incoming_energy += self.neighbors[i].buffer.energy
            incoming_gate += self.neighbors[i].buffer.gate
        self.main.energy += incoming_energy
        self.main.gate += incoming_gate % 8

    @color_effects.MovingAverage
    def color(self):
        """
        Implement the logic of cell's color calculation.

        Here, we highlight cells with valency != 0.
        Color depends on valency value (energy % 8).

        """
        energy = xmath.min(255, self.main.energy)
        vacuum = (energy % 8 == 0) * energy
        red = (((energy % 8) >> 2) & 1) * 255 + vacuum
        blue = (((energy % 8) >> 1) & 1) * 255 + vacuum
        green = ((energy % 8) & 1) * 255 + vacuum
        return (red, green, blue)


class GlyphExperiment(core.Experiment):
    """
    Generate glyph-like scripture.

    Field is filled with small rectangular areas aligned as a grid.
    Each area evolves into a glyph then.

    """

    word = "PSHSHSHSH!"
    size = (640, 360, )
    zoom = 3
    pos = [0, 0]
    fade_in = 1
    fade_out = 1
    smooth_factor = 100
    speed = 200
    seed = seeds.patterns.PrimordialSoup(
        vals={
            "energy": 0,
            "gate": seeds.random.RandInt(0, 2 ** 3 - 1),
            "interval": seeds.random.RandInt(0, 2 ** 12 - 1),
            "rng": seeds.random.RandInt(0, 2 ** 16 - 1),
        }
    )
    for i in range(6 * 3):
        seed += seeds.patterns.BigBang(
            pos=(30 + (i % 6) * 100, 40 + (i // 6) * 100),
            size=(80, 80),
            vals={
                "energy": seeds.random.RandInt(0, 2 ** 7 - 1),
                "gate": seeds.random.RandInt(0, 2 ** 3 - 1),
                "interval": seeds.random.RandInt(0, 2 ** 12 - 1),
                "rng": seeds.random.RandInt(0, 2 ** 16 - 1),
            }
        )


def run():
    """Run model/experiment interactively."""
    moire = importlib.import_module("moire")
    model = PoissonWalk(GlyphExperiment)
    model.speed = 200
    gui = moire.GUI(runnable=model)
    gui.run()


def render():
    """Render a video."""
    mp_editor = importlib.import_module("moviepy.editor")

    def make_frame(_):
        timestep = make_frame.model.timestep
        num_iters = 5000 if timestep < 1200000 else 1000
        if timestep > 1200000 + 120000:
            num_iters = 15000
        for _ in range(num_iters):
            make_frame.model.step()
        width, height = make_frame.model.width, make_frame.model.height
        frame = make_frame.model.render().reshape((height, width, 3))
        return frame

    width, height = GlyphExperiment.size
    zoom = GlyphExperiment.zoom
    width, height = width * zoom, height * zoom
    make_frame.model = PoissonWalk(GlyphExperiment)
    make_frame.model.set_viewport((width, height))

    default_energy = make_frame.model.default_denergy
    animation = mp_editor.VideoClip(make_frame, duration=23)
    animation.write_videofile("poisson_walk_p%d.mp4" % default_energy, fps=24)


if __name__ == "__main__":
    render()
