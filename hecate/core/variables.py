class Variable:
    """
    Base class for all variables.

    """


class IntegerVariable(Variable):
    pass


class DeferredExpression:

    def __init__(self, code):
        self.code = code
