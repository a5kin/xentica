import numpy as np

from .random import LocalRandom


class ValDictMeta(type):
    pass


class ValDict(metaclass=ValDictMeta):
    """
    Wrapper over Python dictionary, to keep descriptors
    along with regular values. READONLY.

    """
    def __init__(self, d, parent=None):
        self._d = d
        self.parent = parent
        if parent is None:
            self.parent = self
            self.random = LocalRandom()

    def items(self):
        for k in self._d.keys():
            v = self[k]
            yield k, v

    def keys(self):
        for k in self._d.keys():
            yield k

    def __getitem__(self, key):
        if key in self._d:
            if hasattr(self._d[key], '__get__'):
                return self._d[key].__get__(self.parent, self.parent.__class__)
            return self._d[key]
        raise KeyError(key)

    def __setitem__(self, key, val):
        raise NotImplementedError


class RandomPattern:
    """
    Base class for random patterns.

    """
    def __init__(self, vals):
        self.random = LocalRandom()
        self.vals = ValDict(vals, self)


class BigBang(RandomPattern):
    """
    Init pattern : small area is initialized with random values.

    """
    def __init__(self, vals, pos=None, size=None):
        self.pos = np.asarray(pos) if pos else None
        self.size = np.asarray(size) if size else None
        super(BigBang, self).__init__(vals)

    def generate(self, cells, cells_num, field_size,
                 index_to_coord, pack_state):
        dims = range(len(field_size))
        if self.size is None:
            rnd_vec = [self.random.std.randint(1, field_size[i]) for i in dims]
            self.size = np.asarray(rnd_vec)
        if self.pos is None:
            rnd_vec = [self.random.std.randint(0, field_size[i]) for i in dims]
            self.pos = np.asarray(rnd_vec)
        for i in range(len(self.pos)):
            x, l, d = self.pos[i], self.size[i], field_size[i]
            if x + l >= d:
                self.pos[i] = d - l
            self.pos[i] = max(0, self.pos[i])
        for i in range(cells_num):
            coord = np.asarray(index_to_coord(i))
            if all(coord >= self.pos) and all(coord < self.pos + self.size):
                state = {}
                for name in sorted(self.vals.keys()):
                    val = self.vals[name]
                    state[name] = val
                cells[i] = pack_state(state)


class PrimordialSoup(RandomPattern):
    """
    Init pattern : the entire field is initialized with random values.

    """
    def generate(self, cells, cells_num, field_size,
                 index_to_coord, pack_state):
        for i in range(cells_num):
            state = {}
            for name in sorted(self.vals.keys()):
                val = self.vals[name]
                state[name] = val
            cells[i] = pack_state(state)
