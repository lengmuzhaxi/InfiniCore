from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

# 注意函数名必须是 erf
def erf(input, *, out=None):
    if out is None:
        return Tensor(_infinicore.erf(input._underlying))
    _infinicore.erf_(out._underlying, input._underlying)

    return out