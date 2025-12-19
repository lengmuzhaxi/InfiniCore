from typing import Optional
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

# 映射 Reduction 字符串到整数 (需与 C++ 后端枚举保持严格一致)
_REDUCTION_MODES = {
    "none": 0,
    "mean": 1,
    "sum": 2,
    "prod": 3,
    "amax": 4,
    "amin": 5,
}

def scatter_reduce(
    input: Tensor, 
    dim: int, 
    index: Tensor, 
    src: Tensor, 
    reduce: str, 
    *, 
    include_self: bool = True, 
    out: Optional[Tensor] = None
) -> Tensor:
    r"""Reduces all values from the `src` tensor to the indices specified in the `index` tensor
    in the `input` tensor.
    """
    
    # 1. 确保输入连续 (Contiguous)
    if not input.is_contiguous():
        input = input.contiguous()
    if not index.is_contiguous():
        index = index.contiguous()
    if not src.is_contiguous():
        src = src.contiguous()

    # 2. 解析 reduction 参数
    if reduce not in _REDUCTION_MODES:
        raise ValueError(f"'{reduce}' is not a valid value for reduction. Supported modes: {list(_REDUCTION_MODES.keys())}")
    reduction_val = _REDUCTION_MODES[reduce]

    # 3. 分发计算
    if out is not None:
        # [关键修复] 处理非连续 Output：使用临时连续 buffer
        # 如果不加这一段，测试集中的 strided output 用例会失败
        if not out.is_contiguous():
            # 1. 创建连续副本 (包含原值，因为 scatter 可能是累加)
            temp_out = out.contiguous()
            
            # 2. 在连续副本上计算
            _infinicore.scatter_reduce_(
                temp_out._underlying, 
                input._underlying, 
                dim,
                index._underlying, 
                src._underlying, 
                reduction_val,
                include_self
            )
            
            # 3. 将结果拷回原 output (copy_ 会自动处理 stride)
            out.copy_(temp_out)
            return out
        
        # 正常连续路径
        _infinicore.scatter_reduce_(
            out._underlying, 
            input._underlying, 
            dim,
            index._underlying, 
            src._underlying, 
            reduction_val,
            include_self
        )
        return out

    # Out-of-place: 调用底层的 functional 接口，返回新 Tensor
    return Tensor(
        _infinicore.scatter_reduce(
            input._underlying, 
            dim,
            index._underlying, 
            src._underlying, 
            reduction_val,
            include_self
        )
    )