"""
Module containing different patterns for CA seed initialization.

Each pattern class having one mandatory method ``generate()`` which is
called automatically at the initialization stage.

Patterns are intended to use in
:class:`Experiment <xentica.core.experiments.Experiment>` classes.
See the example of general usage above.

.. _The Concept: http://artipixoids.a5kin.net/concept/artipixoids_concept.pdf

"""

import abc

import numpy as np

from .random import LocalRandom

__all__ = ['RandomPattern', 'BigBang', 'PrimordialSoup', 'ValDict', ]


class ValDictMeta(type):
    """Placeholder for :class:`ValDict` metaclass."""


class ValDict(metaclass=ValDictMeta):
    """
    Wrapper over Python dictionary.

    It can keep descriptor classes along with regular values. Then, on
    the item getting, the necessary value is automatically obtaining
    either directly or via descriptor logic.

    Readonly, you should set all dictionary values at the class
    initialization.

    Example of usage::

        >>> from xentica.seeds.random import RandInt
        >>> from xentica.seeds.patterns import ValDict
        >>> d = {'a': 2, 's': RandInt(11, 23), 'd': 3.3}
        >>> vd = ValDict(d)
        >>> vd['a']
        2
        >>> vd['s']
        14
        >>> vd['d']
        3.3

    :param d:
        Dictionary with mixed values. May contain descriptor classes.
    :param parent:
        A reference to class holding the dictionary. Optional.

    """

    def __init__(self, d, parent=None):
        """Initialize class."""
        self._d = d
        self.parent = parent
        if parent is None:
            self.parent = self
            self.random = LocalRandom()

    def items(self):
        """Iterate over dictionary items."""
        for k in self._d.keys():
            v = self[k]
            yield k, v

    def keys(self):
        """Iterate over dictionary keys."""
        for k in self._d.keys():
            yield k

    def __getitem__(self, key):
        """Implement the logic of obtaining item from dictionary."""
        if key in self._d:
            if hasattr(self._d[key], '__get__'):
                return self._d[key].__get__(self.parent, self.parent.__class__)
            return self._d[key]
        raise KeyError(key)

    def __setitem__(self, key, val):
        """Suppress direct item setting, may be allowed in future."""
        raise NotImplementedError


class RandomPattern:
    """
    Base class for random patterns.

    :param vals:
        Dictionary with mixed values. May contain descriptor classes.

    """

    def __init__(self, vals):
        """Initialize class."""
        self.random = LocalRandom()
        self.vals = ValDict(vals, self)

    @abc.abstractmethod
    def generate(self, cells, bsca):
        """
        Generate the entire initial state.

        This is an abstract method, you must implement it in
        :class:`RandomPattern` subclasses.

        :param cells:
            NumPy array with cells' states as items. The seed will be
            generated over this array.
        :param cells_num:
            Total number of cells in ``cells`` array.
        :param field_size:
            Tuple with field sizes per each dimension.
        :param index_to_coord:
            Function translating cell's index to coordinate.
        :param pack_state:
            Function packing state into single integer.

        """


class BigBang(RandomPattern):
    """
    Random init pattern, known as *"Big Bang"*.

    Citation from `The Concept`_:

        *"A small area of space is initialized with a high amount of energy
        and random parameters per each quantum. Outside the area, quanta
        has either zero or minimum possible amount of energy. This is a
        good test for the ability of energy to spread in empty space."*

    The current implementation allows to generate a value for every
    cell inside specified N-cube area. Cells outside the area have
    zero values.

    :param vals:
        Dictionary with mixed values. May contain descriptor classes.
    :param pos:
        A tuple with the coordinates of the lowest corner of the Bang area.
    :param size:
        A tuple with the size of Bang area per each dimension.

    """

    def __init__(self, vals, pos=None, size=None):
        """Initialize class."""
        self.pos = np.asarray(pos) if pos else None
        self.size = np.asarray(size) if size else None
        super(BigBang, self).__init__(vals)

    def generate(self, cells, bsca):
        """
        Generate the entire initial state.

        See :meth:`RandomPattern.generate` for details.

        """
        dims = range(len(bsca.size))
        if self.size is None:
            rnd_vec = [self.random.std.randint(1, bsca.size[i]) for i in dims]
            self.size = np.asarray(rnd_vec)
        if self.pos is None:
            rnd_vec = [self.random.std.randint(0, bsca.size[i]) for i in dims]
            self.pos = np.asarray(rnd_vec)
        for i in range(len(self.pos)):
            coord, width, side = self.pos[i], self.size[i], bsca.size[i]
            if coord + width >= side:
                self.pos[i] = side - width
            self.pos[i] = max(0, self.pos[i])
        for i in range(bsca.cells_num):
            coord = np.asarray(bsca.index_to_coord(i))
            if all(coord >= self.pos) and all(coord < self.pos + self.size):
                state = {}
                for name in sorted(self.vals.keys()):
                    val = self.vals[name]
                    state[name] = val
                cells[i] = bsca.pack_state(state)


class PrimordialSoup(RandomPattern):
    """
    Random init pattern, known as *"Primordial Soup"*.

    Citation from `The Concept`_:

        *"Each and every quantum initially has an equally small amount
        of energy, other parameters are random. This is a good test
        for the ability of energy to self-organize in clusters from
        the completely uniform distribution."*

    The current implementation allows to populate the entire board
    with generated values.

    :param vals:
        Dictionary with mixed values. May contain descriptor classes.

    """

    def generate(self, cells, bsca):
        """
        Generate the entire initial state.

        See :meth:`RandomPattern.generate` for details.

        """
        for i in range(bsca.cells_num):
            state = {}
            for name in sorted(self.vals.keys()):
                val = self.vals[name]
                state[name] = val
            cells[i] = bsca.pack_state(state)
