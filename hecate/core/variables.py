import inspect

from cached_property import cached_property


class DeferredExpression:

    def __init__(self, code=''):
        self.code = code
        ops = (
            ('+', 'add'),
            ('-', 'sub'),
            ('*', 'mul'),
            ('/', 'truediv'),
            ('%', 'mod'),
            ('>>', 'rshift'),
            ('<<', 'lshift'),
            ('&', 'and'),
            ('~', 'xor'),
            ('|', 'or'),
        )
        for c_op, base_name in ops:
            def binary_direct(op):
                def op_func(self_var, value):
                    code = "(%s %s %s)" % (self_var, op, value)
                    return DeferredExpression(code)
                return op_func

            def binary_reflected(op):
                def op_func(self_var, value):
                    code = "(%s %s %s)" % (value, op, self_var)
                    return DeferredExpression(code)
                return op_func

            def augmented_assign(op):
                def op_func(self_var, value):
                    if isinstance(self_var, Variable):
                        self_var._declare_once()
                    code = "%s += %s;\n" % (self_var.var_name, value)
                    self_var._bsca._func_body += code
                    return self
                return op_func

            func_name = "__%s__" % base_name
            setattr(self.__class__, func_name, binary_direct(c_op))
            func_name = "__r%s__" % base_name
            setattr(self.__class__, func_name, binary_reflected(c_op))
            func_name = "__i%s__" % base_name
            setattr(self.__class__, func_name, augmented_assign(c_op))

    def __str__(self):
        return self.code


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
