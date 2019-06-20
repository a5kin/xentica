"""
The module containing different patterns for CA seed initialization.

Each pattern class have one mandatory method ``generate()`` which is
called automatically at the initialization stage.

Patterns are intended for use in
:class:`Experiment <xentica.core.experiments.Experiment>` classes.
See the example of general usage above.

.. _The Concept: http://artipixoids.a5kin.net/concept/artipixoids_concept.pdf

"""

import abc

import numpy as np

from .random import LocalRandom

__all__ = ['RandomPattern', 'BigBang', 'PrimordialSoup', 'ValDict', ]


class ValDictMeta(type):
    """A placeholder for :class:`ValDict` metaclass."""


class ValDict(metaclass=ValDictMeta):
    """
    A wrapper over the Python dictionary.

    It can keep descriptor classes along with regular values. When you
    get the item, the necessary value is automatically obtaining
    either directly or via descriptor logic.

    Read-only, you should set all dictionary values at the class
    initialization.

    The example of usage::

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
        A reference to the class holding the dictionary. Optional.

    """

    def __init__(self, d, parent=None):
        """Initialize the class."""
        self._d = d
        self.parent = parent
        if parent is None:
            self.parent = self
            self.random = LocalRandom()

    def items(self):
        """Iterate over dictionary items."""
        for key in self._d.keys():
            val = self[key]
            yield key, val

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
    The base class for random patterns.

    :param vals:
        Dictionary with mixed values. May contain descriptor classes.

    """

    def __init__(self, vals):
        """Initialize the class."""
        self._random = LocalRandom()
        self.vals = ValDict(vals, self)

    @property
    def random(self):
        """Get the random stream."""
        return self._random

    @random.setter
    def random(self, val):
        """Set the random stream."""
        self._random = val

    def __add__(self, other):
        """Return a chained pattern."""
        return ChainedPattern(self, other)

    @abc.abstractmethod
    def generate(self, cells, bsca):
        """
        Generate the entire initial state.

        This is an abstract method, you must implement it in
        :class:`RandomPattern` subclasses.

        :param cells:
            NumPy array with cells' states as items. The seed will be
            generated over this array.
        :param bsca:
            :class:`xentica.core.CellularAutomaton` instance, to access
            the field's size and other attributes.

        """


class ChainedPattern(RandomPattern):
    """The join of two other patterns."""

    def __init__(self, pattern1, pattern2):
        self._pattern1 = pattern1
        self._pattern2 = pattern2
        super(ChainedPattern, self).__init__({})

    @RandomPattern.random.setter
    def random(self, val):
        """Set the random stream."""
        self._pattern1.random = val
        self._pattern2.random = val

    def generate(self, cells, bsca):
        """
        Generate two patterns sequentially.

        See :meth:`RandomPattern.generate` for details.

        """
        self._pattern1.generate(cells, bsca)
        self._pattern2.generate(cells, bsca)


class BigBang(RandomPattern):
    """
    Random init pattern, known as *"Big Bang"*.

    Citation from `The Concept`_:

        *"A small area of space is initialized with a high amount of energy
        and random parameters per each quantum. Outside the area, quanta
        has either zero or minimum possible amount of energy. This is a
        good test for the ability of energy to spread in empty space."*

    The current implementation generates a value for every cell inside a
    specified N-cube area. Cells outside the area remain unchanged.

    :param vals:
        Dictionary with mixed values. May contain descriptor classes.
    :param pos:
        A tuple with the coordinates of the lowest corner of the Bang area.
    :param size:
        A tuple with the size of the Bang area per each dimension.

    """

    def __init__(self, vals, pos=None, size=None):
        """Initialize class."""
        self.pos = np.asarray(pos) if pos else None
        self.size = np.asarray(size) if size else None
        super(BigBang, self).__init__(vals)

    def _prepare_area(self, bsca_size):
        """
        Prepare area size and position.

        :param bsca_size: tuple with CA size.

        """
        dims = range(len(bsca_size))
        randint = self.random.standard.randint
        if self.size is None:
            rnd_vec = [randint(1, bsca_size[i]) for i in dims]
            self.size = np.asarray(rnd_vec)
        if self.pos is None:
            rnd_vec = [randint(0, bsca_size[i]) for i in dims]
            self.pos = np.asarray(rnd_vec)
        for i in range(len(self.pos)):
            coord, width, side = self.pos[i], self.size[i], bsca_size[i]
            if coord + width >= side:
                self.pos[i] = side - width
            self.pos[i] = max(0, self.pos[i])

    def generate(self, cells, bsca):
        """
        Generate the entire initial state.

        See :meth:`RandomPattern.generate` for details.

        """
        self._prepare_area(bsca.size)
        indices = np.arange(0, bsca.cells_num)
        coords = bsca.index_to_coord(indices)
        area = None
        for i in range(len(bsca.size)):
            condition = (coords[i] >= self.pos[i])
            condition &= (coords[i] < self.pos[i] + self.size[i])
            if area is None:
                area = condition
                continue
            area &= (condition)
        state = {}
        for name in sorted(self.vals.keys()):
            val = self.vals[name]
            state[name] = val
        cells[np.where(area)] = bsca.pack_state(state)


class PrimordialSoup(RandomPattern):
    """
    Random init pattern, known as *"Primordial Soup"*.

    Citation from `The Concept`_:

        *"Each and every quantum initially has an equally small amount
        of energy, other parameters are random. This is a good test
        for the ability of energy to self-organize in clusters from
        the completely uniform distribution."*

    The current implementation populates the entire board with
    generated values.

    :param vals:
        Dictionary with mixed values. May contain descriptor classes.

    """

    def __init__(self, vals):
        """Initialize class."""
        self.size = None
        super(PrimordialSoup, self).__init__(vals)

    def generate(self, cells, bsca):
        """
        Generate the entire initial state.

        See :meth:`RandomPattern.generate` for details.

        """
        self.size = bsca.size
        state = {}
        for name in sorted(self.vals.keys()):
            val = self.vals[name]
            state[name] = val
        cells[:bsca.cells_num] = bsca.pack_state(state)
