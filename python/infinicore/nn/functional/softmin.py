import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

def softmin(input: Tensor, dim: int, dtype=None, *, out=None) -> Tensor:
    r"""Computes the softmin function value for the given input tensor."""
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        # 根据之前定义的 ntops 接口，这里传递 input, dim, dtype
        return infinicore.ntops.torch.softmin(input, dim, dtype=dtype)

    if out is None:
        return Tensor(_infinicore.softmin(input._underlying, dim, dtype))

    # 假设底层 C++ 绑定提供了带 out 输出的 inplace/out 变体 (通常以 _ 结尾)
    _infinicore.softmin_(out._underlying, input._underlying, dim, dtype)
    return out