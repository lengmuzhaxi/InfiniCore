from typing import Optional
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

def pairwise_distance(
    x1: Tensor, 
    x2: Tensor, 
    p: float = 2.0, 
    eps: float = 1e-6, 
    keepdim: bool = False,  # <--- 修改 1: 新增参数
    *, 
    out: Optional[Tensor] = None
) -> Tensor:
    r"""Computes the pairwise distance between vectors v1, v2 using the p-norm.
    """
    
    if out is not None:
        _infinicore.pairwise_distance_(
            out._underlying,
            x1._underlying,
            x2._underlying,
            p,
            eps,
            keepdim  # <--- 修改 3: 传递 keepdim
        )
        return out

    return Tensor(
        _infinicore.pairwise_distance(
            x1._underlying,
            x2._underlying,
            p,
            eps,
            keepdim  # <--- 修改 4: 传递 keepdim
        )
    )