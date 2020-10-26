"""
https://numpy.org/doc/stable/user/basics.subclassing.html
"""
from operator import mul
from functools import reduce
from collections import UserDict
from typing import Union, Tuple, List

import numpy as np

AGG_FUNCTIONS = ['all', 'any', 'argmax', 'argmin', 'cumprod', 'cumsum',
                 'max', 'mean', 'min', 'prod', 'std', 'sum', 'var']


class NamedArray(np.ndarray):

    def __new__(cls,
                array: np.ndarray,
                axis_names: str,
                aliases: dict = None):
        print("__new__ was called")
        obj = np.asarray(array).view(cls)
        obj.axis_names: dict = {
            name: i for i, name in enumerate(axis_names.split(","))}
        obj.aliases = aliases

        return obj

    # pylint:disable=attribute-defined-outside-init
    def __array_finalize__(self, obj: np.ndarray):
        """
        Based on the numpy docs, this is the only method that always
        sees new instances being created. Hence, it is a sensible
        place to fill in instance defaults for new object attributes.
        """
        print("__array_finalize__ was called")
        if obj is None:
            return

        self._obj = obj
        self.axis_names = getattr(obj, "axis_names", None)
        self.aliases = getattr(obj, "aliases", {})
    # pylint:enable=attribute-defined-outside-init

    ############
    # indexing #
    ############

    def __call__(self, **kwargs):
        indexer = [slice(None)]*self._obj.ndim
        for name, idx in kwargs.items():
            indexer[self.axis_names[name]] = idx
        indexer = tuple(indexer)
        self._obj = self._obj[indexer]
        return self

    ################
    # manipulation #
    ################

    def transpose(self, expr: str):
        """[summary]

        Args:
            expr (str): [description]

        Returns:
            [type]: [description]
        """
        axis = self._expr2axis(expr)
        # TODO change self.axis_names
        self._obj = self._obj.transpose(axis)
        return self

    def reshape(self, expr: str, order="C"):
        """Returns an array containing the same data with a new shape.

        Note:
            The np.ndarray.reshape does not work this way. Instead you have to
            specify the new shape. Here you only specify the names of the axes.

        Args:
            expr (str): [description]

        Returns:
            [type]: [description]
        """
        axis = []
        shape = []
        aliases = {}

        for name in expr.split(","):
            if name in self.axis_names:
                axis.append(name)
                sz = self.shape[self.axis_names[name]]
            else:
                new_name, old_names = name.split("=")
                axis.append(new_name)
                sz = reduce(mul, [self.shape[self.axis_names[n]]
                                  for n in old_names.split("*")])
                aliases[new_name] = old_names
            shape.append(sz)

        obj = self._obj.reshape(shape, order=order)
        axis = ",".join(list(axis))

        return NamedArray(obj, axis, aliases)

    def swapaxes(self, axis1: str, axis2: str):
        axis1: int = self.axis_names[axis1]
        axis2: int = self.axis_names[axis2]

        # TODO change self.axis_names

        self._obj = self._obj.swapaxes(axis1, axis2)
        return self

    def squeeze(self, axis: str = None):
        names = dict(self.axis_names)
        if axis is not None:
            axis: int = names[axis]
            for name, idx in names.items():
                if self._obj.shape[idx] == 1 and name in axis:
                    del names[name]
        
        _obj = self._obj.squeeze(axis=axis)
        names = ",".join(list(names))

        return NamedArray(_obj, names)

    def collapse(self, expr: str):
        """Similar to reshape"""

    def expand(self, axis: str):
        """Similar to reshape"""
    
    ##########
    # reduce #
    ##########

    def argmax(self, axis=None, out=None):
        raise NotImplementedError

    def argmin(self, axis=None, out=None):
        raise NotImplementedError

    def all(self, axis=None, out=None, keepdims=False):
        raise NotImplementedError

    def any(self, axis=None, out=None, keepdims=False):
        raise NotImplementedError

    def cumprod(self, axis=None, dtype=None, out=None):
        raise NotImplementedError

    def cumsum(self, axis=None, dtype=None, out=False):
        raise NotImplementedError

    def sum(self, axis=None, dtype=None, out=None, keepdims=False,
            initial=0, where=True):
        raise NotImplementedError

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        raise NotImplementedError

    def max(self, axis=None, out=None, keepdims=False, initial=None,
            where=True):
        names = dict(self.axis_names)

        # TODO
        if not keepdims:
            del names[axis]
            names = ",".join(list(names))

        axis = self._expr2axis(axis)
        obj = self._obj.max(axis=axis, out=out, keepdims=keepdims,
                            initial=initial, where=where)

        return NamedArray(obj, names)

    def min(self, axis=None, out=None, keepdims=False, initial=None,
            where=True):
        raise NotImplementedError

    def prod(self, axis=None, dtype=None, out=None, keepdims=False,
             initial=1, where=True):
        raise NotImplementedError

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        named_axis = self._expr2axis(axis)
        obj = self._obj.std(axis=named_axis, dtype=dtype,
                            out=out, ddof=0, keepdims=keepdims)
        return obj

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        raise NotImplementedError

    ###

    def __matmul__(self, rhs):
        if isinstance(rhs, NamedArray):
            raise NotImplementedError
        else:
            print("Print some warning here")

    ##########
    # utils #
    ##########

    def axis2name(self):
        pass

    def _expr2axis(self, expr: Union[str, Tuple[str]], named=False) -> tuple:
        if expr is None:
            return None

        if isinstance(expr, str):
            axis = tuple(self.axis_names[name] for name in expr.split(","))
        elif isinstance(expr, tuple):
            axis = tuple(self.axis_names[name] for name in expr)
        else:
            raise ValueError

        return axis

    def _check_axis_names(self):
        pass

    def _remove_axes(self, axes_to_remove: List[str]) -> dict:
        names = dict(self.axis_names)
        for axis in axes_to_remove:
            if axis in names:
                del names[axis]
        return names

    def to_numpy(self):
        return self._obj


class NamedAxis(UserDict):
    def swapaxes(self, axis1, axis2):
        pass

    def remove(self, *axes):
        pass

    def reshape(self):
        pass

    def squeeze(self):
        pass


"""
       str: "b,c,h,w"
Tuple[str]: ("b", "c", "h", "w")
      dict: {"b": 0, "c": 1, ...}
"""
