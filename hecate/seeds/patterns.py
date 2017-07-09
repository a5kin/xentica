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
        if self.size is None:
            # TODO: size / pos interdependence
            self.size = np.asarray((self.random.std.randint(1, field_size[0]),
                                    self.random.std.randint(1, field_size[1])))
        if self.pos is None:
            # TODO: size / pos interdependence
            self.pos = np.asarray((self.random.std.randint(1, field_size[0]),
                                   self.random.std.randint(1, field_size[1])))
        for i in range(cells_num):
            coord = np.asarray(index_to_coord(i))
            if all(coord >= self.pos) and all(coord < self.pos + self.size):
                state = {}
                for name, val in self.vals.items():
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
            for name, val in self.vals.items():
                state[name] = val
            cells[i] = pack_state(state)
