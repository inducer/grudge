__copyright__ = """
Copyright (C) 2013 Andreas Kloeckner
Copyright (C) 2019 Matt Wala
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import numpy as np

from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa
from pytools import RecordWithoutPickling


# {{{ helpers

def should_use_numpy(arg):
    from numbers import Number
    if isinstance(arg, Number) or \
            isinstance(arg, np.ndarray) and arg.shape == ():
        return True
    return False


def cl_to_numpy_function_name(name):
    return {
        "atan2": "arctan2",
        }.get(name, name)

# }}}


# {{{ function

class FunctionNotFound(KeyError):
    pass


class Function(RecordWithoutPickling):
    """
    .. attribute:: identifier
    .. attribute:: supports_codegen
    .. automethod:: __call__
    .. automethod:: get_result_dofdesc
    """

    def __init__(self, identifier, **kwargs):
        super().__init__(identifier=identifier, **kwargs)

    def __call__(self, queue, *args, **kwargs):
        """Call the function implementation, if available."""
        raise TypeError("function '%s' is not callable" % self.identifier)

    def get_result_dofdesc(self, arg_dds):
        """Return the :class:`grudge.symbolic.primitives.DOFDesc` for the return value
        of the function.

        :arg arg_dds: A list of :class:`grudge.symbolic.primitives.DOFDesc` instances
            for each argument
        """
        raise NotImplementedError


class CElementwiseFunction(Function):
    supports_codegen = True

    def __init__(self, identifier, nargs):
        super().__init__(identifier=identifier, nargs=nargs)

    def get_result_dofdesc(self, arg_dds):
        assert len(arg_dds) == self.nargs
        return arg_dds[0]

    def __call__(self, array_context, *args):
        func_name = self.identifier
        from pytools import single_valued
        if single_valued(should_use_numpy(arg) for arg in args):
            func = getattr(np, func_name)
            return func(*args)

        if func_name == "fabs":  # FIXME
            # Loopy has a type-adaptive "abs", but no "fabs".
            func_name = "abs"

        sfunc = getattr(array_context.np, func_name)
        return sfunc(*args)


class CBesselFunction(Function):

    supports_codegen = True

    def get_result_dofdesc(self, arg_dds):
        assert len(arg_dds) == 2
        return arg_dds[1]


class FixedDOFDescExternalFunction(Function):

    supports_codegen = False

    def __init__(self, identifier, implementation, dd):
        super().__init__(
                identifier,
                implementation=implementation,
                dd=dd)

    def __call__(self, array_context, *args, **kwargs):
        return self.implementation(array_context, *args, **kwargs)

    def get_result_dofdesc(self, arg_dds):
        return self.dd

# }}}


# {{{ function registry

class FunctionRegistry(RecordWithoutPickling):
    def __init__(self, id_to_function=None):
        if id_to_function is None:
            id_to_function = {}

        super().__init__(id_to_function=id_to_function)

    def register(self, function):
        """Return a copy of *self* with *function* registered."""

        if function.identifier in self.id_to_function:
            raise ValueError("function '%s' is already registered"
                    % function.identifier)

        new_id_to_function = self.id_to_function.copy()
        new_id_to_function[function.identifier] = function
        return self.copy(id_to_function=new_id_to_function)

    def __getitem__(self, function_id):
        try:
            return self.id_to_function[function_id]
        except KeyError:
            raise FunctionNotFound(
                    "unknown function: '%s'"
                    % function_id)

    def __contains__(self, function_id):
        return function_id in self.id_to_function

# }}}


def _make_bfr():
    bfr = FunctionRegistry()

    bfr = bfr.register(CElementwiseFunction("sqrt", 1))
    bfr = bfr.register(CElementwiseFunction("exp", 1))
    bfr = bfr.register(CElementwiseFunction("fabs", 1))
    bfr = bfr.register(CElementwiseFunction("sin", 1))
    bfr = bfr.register(CElementwiseFunction("cos", 1))
    bfr = bfr.register(CElementwiseFunction("atan2", 2))
    bfr = bfr.register(CBesselFunction("bessel_j"))
    bfr = bfr.register(CBesselFunction("bessel_y"))

    return bfr


base_function_registry = _make_bfr()


def register_external_function(
        function_registry, identifier, implementation, dd):
    return function_registry.register(
            FixedDOFDescExternalFunction(
                identifier, implementation, dd))

# vim: foldmethod=marker
