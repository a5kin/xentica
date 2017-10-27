import inspect

from cached_property import cached_property


class DeferredExpression:

    def __init__(self, code=''):
        self.code = code

    def __str__(self):
        return self.code

    def __rshift__(self, value):
        code = "(%s >> %s)" % (self, value)
        return DeferredExpression(code)

    def __rrshift__(self, value):
        code = "(%s >> %s)" % (value, self)
        return DeferredExpression(code)

    def __and__(self, value):
        code = "(%s & %s)" % (self, value)
        return DeferredExpression(code)

    def __rand__(self, value):
        code = "(%s & %s)" % (value, self)
        return DeferredExpression(code)

    def __or__(self, value):
        code = "(%s | %s)" % (self, value)
        return DeferredExpression(code)

    def __ror__(self, value):
        code = "(%s | %s)" % (value, self)
        return DeferredExpression(code)

    def __iadd__(self, value):
        if isinstance(self, Variable):
            self._declare_once()
        code = "%s += %s;\n" % (self.var_name, value)
        self._bsca._func_body += code
        return self


class Variable(DeferredExpression):
    """
    Base class for all variables.

    """
    def __init__(self, val=None):
        super(Variable, self).__init__()
        self._declared = False
        if val is not None:
            self._init_val = DeferredExpression(str(val))

    @property
    def _holder_frame(self):
        frame = inspect.currentframe().f_back.f_back.f_back
        while isinstance(frame.f_locals.get('self', ''), Variable):
            frame = frame.f_back
        return frame

    @cached_property
    def _bsca(self):
        return self._holder_frame.f_locals.get('self', '')

    @cached_property
    def var_name(self):
        all_vars = self._holder_frame.f_locals.items()
        for k, var in all_vars:
            if isinstance(var, self.__class__):
                if hash(self) == hash(var):
                    return k
        return ''

    def _declare_once(self):
        if not self._declared:
            c = "%s %s = %s;\n" % (
                self.var_type, self.var_name, self._init_val
            )
            self._bsca._func_body += c
            self._declared = True
            setattr(self._bsca, self.var_name, self)

    def __str__(self):
        return self.var_name

    def __get__(self, obj, objtype):
        self._declare_once()
        return self

    def __set__(self, obj, value):
        self._declare_once()
        code = "%s = %s;\n" % (self.var_name, value)
        self._bsca._func_body += code


class IntegerVariable(Variable):
    var_type = "unsigned int"

    def __init__(self, val=0):
        super(IntegerVariable, self).__init__(val)
