import random


class RandInt:

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __get__(self, obj, objtype):
        return random.randint(self.min_val, self.max_val)
