import numpy as np

from .random import LocalRandom


class ValDictMeta(type):
    pass


class ValDict(metaclass=ValDictMeta):
    """
    Wrapper over Python dictionary, to keep descriptors
    along with regular values. READONLY.

    """
    def __init__(self, d):
        for k, v in d.items():
            setattr(self.__class__, k, v)
        self.__class__.random = LocalRandom()  # TODO: get RNG from owner

    def items(self):
        for k, v in self.__class__.__dict__.items():
            v = getattr(self.__class__, k)
            yield k, v

    def __getitem__(self, key):
        return getattr(self.__class__, key)

    def __setitem__(self, key, val):
        raise NotImplementedError


class BigBang:
    """
    One of random init patterns described in The Concept.

    It's unclear yet, should we inherit it from some base pattern class.

    """
    def __init__(self, vals, pos=None, size=None):
        self.pos = np.asarray(pos)
        self.size = np.asarray(size)
        self.random = LocalRandom()
        self.vals = ValDict(vals)

    def generate(self, cells, cells_num, index_to_coord, pack_state):
        for i in range(cells_num):
            coord = np.asarray(index_to_coord(i))
            if all(coord >= self.pos) and all(coord < self.pos + self.size):
                state = {}
                for name, val in self.vals.items():
                    state[name] = val
                cells[i] = pack_state(state)
