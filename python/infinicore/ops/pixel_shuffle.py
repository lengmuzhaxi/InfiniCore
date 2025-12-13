from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

def pixel_shuffle(input, upscale_factor, *, out=None):
    # 1. Out-of-place 模式 (如果没有指定 out)
    if out is None:
        return Tensor(_infinicore.pixel_shuffle(
            input._underlying,
            upscale_factor
        ))

    # 2. In-place 模式 (指定了 out)
    # 注意: PixelShuffle 会改变形状，传入的 out 必须形状正确
    _infinicore.pixel_shuffle_(
        out._underlying,
        input._underlying,
        upscale_factor
    )

    return out