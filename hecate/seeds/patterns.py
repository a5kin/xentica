import numpy as np

from .random import LocalRandom


class BigBang:
    """
    One of random init patterns described in The Concept.

    It's unclear yet, should we inherit it from some base pattern class.

    """
    def __init__(self, vals, pos=None, size=None):
        self.pos = np.asarray(pos)
        self.size = np.asarray(size)
        self.vals = vals
        self.random = LocalRandom()

    def generate(self, cells, cells_num, index_to_coord, pack_state):
        for i in range(cells_num):
            coord = np.asarray(index_to_coord(i))
            if all(coord >= self.pos) and all(coord < self.pos + self.size):
                state = {}
                for name, val in self.vals.items():
                    state[name] = val
                cells[i] = pack_state(state)
