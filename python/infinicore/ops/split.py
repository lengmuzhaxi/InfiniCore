from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

def split(input, split_size_or_sections, dim=0, *, out=None):
    if out is None:
        impls = _infinicore.split(input._underlying, split_size_or_sections, dim)
        return [Tensor(impl) for impl in impls]
    out_impls = [t._underlying for t in out]
    _infinicore.split_(out_impls, input._underlying, dim)

    return out