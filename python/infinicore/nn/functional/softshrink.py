import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

def softshrink(input: Tensor, lambd: float = 0.5, *, out=None) -> Tensor:
    r"""Computes the softshrink function value for the given input tensor."""
    
    # 1. 检查是否走 ntops 加速
    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        
        op = infinicore.ntops.torch.softshrink
        
        # === 终极暴力兼容法 (Try-Except) ===
        # 不管 op 是函数、模块，还是模块里的模块，尝试直到成功为止
        try:
            # 尝试 1: 假设它是函数 (Function)
            return op(input, lambd)
        except TypeError:
            # 报错 'module object is not callable'，说明 op 是模块
            try:
                # 尝试 2: 假设它是模块，取内部同名属性 (Module.function)
                return op.softshrink(input, lambd)
            except TypeError:
                # 居然还在报错？说明 op.softshrink 还是模块！
                # 尝试 3: 再剥一层 (Package.Module.function)
                return op.softshrink.softshrink(input, lambd)

    # 2. CPU 或 out 不为 None 的情况
    if out is None:
        return Tensor(_infinicore.softshrink(input._underlying, lambd))

    # 3. 原地操作
    _infinicore.softshrink_(out._underlying, input._underlying, lambd)
    return out