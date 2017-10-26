import inspect
import itertools

from cached_property import cached_property


class Variable:
    """
    Base class for all variables.

    """
    def __init__(self, val=None):
        self._declared = False
        if val is not None:
            self.__set__(self, DeferredExpression(str(val)))

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
        all_vars = itertools.chain(self._holder_frame.f_globals.items(),
                                   self._holder_frame.f_locals.items())
        for k, var in all_vars:
            if isinstance(var, self.__class__):
                if hash(self) == hash(var):
                    return k
        return ''

    def _declare_once(self):
        if not self._declared:
            code = "%s %s;\n" % (self.var_type, self.var_name)
            self._bsca._func_body += code
            self._declared = True

    def __get__(self, obj, objtype):
        self._declare_once()
        return DeferredExpression(self.var_name)

    def __set__(self, obj, value):
        self._declare_once()
        code = "%s = %s;\n" % (self.var_name, value.code)
        self._bsca._func_body += code


class IntegerVariable(Variable):
    var_type = "unsigned int"


class DeferredExpression:

    def __init__(self, code):
        self.code = code
