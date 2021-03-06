# NumPy named tensors

Tensor manipulation in NumPy can be pretty troublesome
if you are dealing with multiple dimensions. 
Here I introduce a manner to manipulate tensors using named axes.

---

First we create a NumPy array.

```python
>>> import numpy as np
>>> x = np.random.randint(0, 10, (2, 3, 4, 4))
```

Create a NamedTensor object.

```python
>>> from nnt import NamedTensor
>>> img = NamedTensor(x, "b,c,h,w")
```

Swap the height `h` and width `w`:

```python
>>> img.transpose("c,w,h,b")
```

For every image `c,h,w`, get the mean across examples in the batch `b`.

```python
>>> img.mean("b")
```

Examples

| NumPy     | NamedTensor     |
|-----------|-----------------|
| `x.mean(0)` | `x.mean("batch")` |
|           |                 |
|           |                 |
