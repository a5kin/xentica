"""Tests for ``xentica.seeds.random`` module."""
import unittest

from xentica.seeds.random import RandInt, LocalRandom


TEST_SEED = "test"


class TestLocalRandom(unittest.TestCase):
    """Tests for ``LocalRandom`` class."""

    def test_standard_random(self):
        """Test built-in random works."""
        rng = LocalRandom()
        val = rng.std.random()
        for _ in range(10):
            new_val = rng.std.random()
            self.assertNotEqual(val, new_val,
                                "Not a random sequence: numbers repeating.")
            val = new_val

    def test_numpy_rand(self):
        """Test ``numpy`` random works."""
        rng = LocalRandom()
        val = rng.np.rand(11,)
        for i in range(10):
            self.assertNotEqual(val[i], val[i + 1],
                                "Not a random sequence: numbers repeating.")

    def test_standard_random_seed(self):
        """Test built-in random with seed."""
        rng = LocalRandom(TEST_SEED)
        sequence_valid = [22, 15, 21, 23, 14, 14, 11, 20, 17, 23]
        sequence_generated = [rng.std.randint(11, 23) for i in range(10)]
        self.assertListEqual(sequence_valid, sequence_generated,
                             "Wrong sequence for seed '%s'." % TEST_SEED)

    def test_numpy_rand_seed(self):
        """Test ``numpy`` random with seed."""
        rng = LocalRandom(TEST_SEED)
        sequence_valid = [19, 16, 22, 19, 15, 22, 20, 20, 13, 15]
        sequence_generated = list(rng.np.randint(11, 23, (10, )))
        self.assertListEqual(sequence_valid, sequence_generated,
                             "Wrong sequence for seed '%s'." % TEST_SEED)


class RandomHolder:
    """Helper to hold ``RandInt``."""

    def __init__(self, seed=None):
        """Initialize ``LocalRandom``."""
        self.random = LocalRandom(seed)


class TestRandInt(unittest.TestCase):
    """Tests for ``RandInt`` class."""

    def test_interval(self):
        """Test generated values are in valid range."""
        holder = RandomHolder()
        holder.__class__.rand_val = RandInt(11, 23)
        for _ in range(10):
            self.assertGreaterEqual(holder.rand_val, 11, "Wrong random value.")
            self.assertLessEqual(holder.rand_val, 23, "Wrong random value.")

    def test_seed(self):
        """Test correct sequence in generated with a seed."""
        holder = RandomHolder(TEST_SEED)
        holder.__class__.rand_val = RandInt(11, 23)
        sequence_valid = [22, 15, 21, 23, 14, 14, 11, 20, 17, 23]
        sequence_generated = [holder.rand_val for i in range(10)]
        self.assertListEqual(sequence_valid, sequence_generated,
                             "Wrong sequence for seed '%s'." % TEST_SEED)
