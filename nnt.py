import numpy as np


class NamedTensor:
    def __init__(self, obj: np.ndarray, names: str):
        self.obj: np.ndarray = obj
        self.names: dict = {
            name: i for i, name in enumerate(names.split(","))}

    def __repr__(self):
        repre1 = " ".join(list(self.names))
        repre2 = self.obj.__repr__()
        return repre1 + "\n" + repre2

    def forevery(self, expr: str):
        return namedtensorop(self.obj, self.names, expr)

    def transpose(self, expr: str):
        axes = tuple(self.names[name] for name in expr.split(","))
        obj = self.obj.transpose(axes)
        return NamedTensor(obj, expr)

    def reshape(self, expr: str):
        axes = tuple(self.names[name] for name in expr.split(","))

    def transform(self, expr: str):
        return

    def __getattr__(self, attr: str):
        if attr == "shape":
            return self.obj.shape

    def __getitem__(self, *args, **kwargs):
        return self.obj.__getitem__(*args, **kwargs)

    def __call__(self, **kwargs):
        indexer = [slice(None)]*self.obj.ndim
        for name, idx in kwargs.items():
            indexer[self.names[name]] = idx
        indexer = tuple(indexer)
        return self.obj[indexer]


class namedtensorop:
    def __init__(self, obj: np.ndarray, names: str, anchor_expr: str):
        self.obj = obj
        self.names = names
        self.anchor_expr = anchor_expr

    def mean(self, expr: str, *args, **kwargs):
        axes = tuple([self.names[name] for name in expr.split(",")])
        obj = self.obj.mean(axis=axes, *args, **kwargs)
        return NamedTensor(obj, self.anchor_expr)
