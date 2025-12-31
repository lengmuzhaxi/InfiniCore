from typing import Optional
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

_REDUCTION_MODES = {
    "none": 0,
    "mean": 1,
    "sum": 2,
}

def margin_ranking_loss(
    input1: Tensor, 
    input2: Tensor, 
    target: Tensor, 
    margin: float = 0.0, 
    reduction: str = "mean", 
    *, 
    out: Optional[Tensor] = None
) -> Tensor:
    r"""Creates a criterion that measures the loss given inputs x1, x2, two 1D mini-batch Tensors,
    and a label 1D mini-batch tensor y (containing 1 or -1).
    """

    if not input1.is_contiguous():
        input1 = input1.contiguous()
    if not input2.is_contiguous():
        input2 = input2.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()
    
    # 解析 reduction 参数
    if reduction not in _REDUCTION_MODES:
        raise ValueError(f"{reduction} is not a valid value for reduction")
    reduction_val = _REDUCTION_MODES[reduction]

    if out is not None:
        _infinicore.margin_ranking_loss_(
            out._underlying,
            input1._underlying,
            input2._underlying,
            target._underlying,
            margin,
            reduction_val
        )
        return out

    return Tensor(
        _infinicore.margin_ranking_loss(
            input1._underlying,
            input2._underlying,
            target._underlying,
            margin,
            reduction_val
        )
    )