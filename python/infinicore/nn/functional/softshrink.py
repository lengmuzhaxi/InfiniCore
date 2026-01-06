from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def softshrink(input, lambd=0.5, *, out=None):
    if out is None:
        return Tensor(_infinicore.softshrink(input._underlying, float(lambd)))

    _infinicore.softshrink_(out._underlying, input._underlying, float(lambd))

    return out
