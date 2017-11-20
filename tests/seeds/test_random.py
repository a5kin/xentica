import unittest

from xentica.seeds.random import RandInt, LocalRandom


TEST_SEED = "test"


class TestLocalRandom(unittest.TestCase):

    def test_standard_random(self):
        rng = LocalRandom()
        val = rng.std.random()
        for i in range(10):
            new_val = rng.std.random()
            self.assertNotEqual(val, new_val,
                                "Not a random sequence: numbers repeating.")
            val = new_val

    def test_numpy_rand(self):
        rng = LocalRandom()
        val = rng.np.rand(11,)
        for i in range(10):
            self.assertNotEqual(val[i], val[i + 1],
                                "Not a random sequence: numbers repeating.")

    def test_standard_random_seed(self):
        rng = LocalRandom(TEST_SEED)
        sequence_valid = [22, 15, 21, 23, 14, 14, 11, 20, 17, 23]
        sequence_generated = [rng.std.randint(11, 23) for i in range(10)]
        self.assertListEqual(sequence_valid, sequence_generated,
                             "Wrong sequence for seed '%s'." % TEST_SEED)

    def test_numpy_rand_seed(self):
        rng = LocalRandom(TEST_SEED)
        sequence_valid = [19, 16, 22, 19, 15, 22, 20, 20, 13, 15]
        sequence_generated = list(rng.np.randint(11, 23, (10, )))
        self.assertListEqual(sequence_valid, sequence_generated,
                             "Wrong sequence for seed '%s'." % TEST_SEED)


class RandomHolder:

    def __init__(self, seed=None):
        self.random = LocalRandom(seed)


class TestRandInt(unittest.TestCase):

    def test_interval(self):
        holder = RandomHolder()
        holder.__class__.rand_val = RandInt(11, 23)
        for i in range(10):
            self.assertGreaterEqual(holder.rand_val, 11, "Wrong random value.")
            self.assertLessEqual(holder.rand_val, 23, "Wrong random value.")

    def test_seed(self):
        holder = RandomHolder(TEST_SEED)
        holder.__class__.rand_val = RandInt(11, 23)
        sequence_valid = [22, 15, 21, 23, 14, 14, 11, 20, 17, 23]
        sequence_generated = [holder.rand_val for i in range(10)]
        self.assertListEqual(sequence_valid, sequence_generated,
                             "Wrong sequence for seed '%s'." % TEST_SEED)
