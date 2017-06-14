from .random import LocalRandom


class BigBang:
    """
    One of random init patterns described in The Concept.

    It's unclear yet, should we inherit it from some base pattern class.

    """
    def __init__(self, pos, size, vals):
        self.pos = pos
        self.size = size
        self.vals = vals
        self.random = LocalRandom()

    def generate(self, cells, field_size):
        cells += self.random.np.randint(2, size=cells.shape, dtype=cells.dtype)
