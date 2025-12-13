from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

def matrix_power(input, n, *, out=None):
    # 1. Out-of-place 模式 (如果没有指定 out)
    if out is None:
        return Tensor(_infinicore.matrix_power(
            input._underlying,
            n
        ))

    # 2. In-place 模式 (指定了 out)
    _infinicore.matrix_power_(
        out._underlying,
        input._underlying,
        n
    )

    return out