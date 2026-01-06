from typing import Optional, Union, Sequence, List
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

def upsample_nearest(
    input: Tensor,
    size: Optional[Union[int, Sequence[int]]] = None,
    scale_factor: Optional[Union[float, Sequence[float]]] = None,
    *,
    out: Optional[Tensor] = None
) -> Tensor:
    r"""
    Applies nearest neighbor upsampling to the input tensor.
    """
    if not input.is_contiguous():
        input = input.contiguous()

    if (size is None) == (scale_factor is None):
        raise ValueError("Either size or scale_factor should be defined, but not both.")

    ndim = len(input.shape)
    output_size = []
    
    # 1. 处理 size 参数
    if size is not None:
        if isinstance(size, int):
            # 如果是单个整数，根据维度广播
            if ndim == 3: # (N, C, W)
                output_size = [size]
            else: # 默认为 2D spatial (N, C, H, W)
                output_size = [size, size]
        elif isinstance(size, (list, tuple)):
            output_size = [int(s) for s in size]
        else:
            raise ValueError("size must be int or sequence of int")
            
    # 2. 处理 scale_factor 参数
    else:
        # 修正: 允许 int 和 float 类型
        scales = []
        if isinstance(scale_factor, (float, int)):
            scales = [float(scale_factor)]
        elif isinstance(scale_factor, (list, tuple)):
            scales = [float(s) for s in scale_factor]
        else:
            raise ValueError("scale_factor must be float or sequence of float")
        
        # 根据维度计算具体的 output_size
        if ndim == 3:
            # Case: [N, C, W] -> Output [W_new]
            w_in = input.shape[-1]
            # 取第一个 scale 因子，如果只有一个则直接使用，如果有多个取最后一个作为 W 的缩放
            scale_w = scales[0] if len(scales) == 1 else scales[-1]
            output_size = [int(w_in * scale_w)]
        else:
            # Case: [N, C, H, W] -> Output [H_new, W_new]
            if len(scales) == 1:
                scale_h = scales[0]
                scale_w = scales[0]
            elif len(scales) >= 2:
                scale_h = scales[0]
                scale_w = scales[1]
            else:
                 raise ValueError("scale_factor sequence length mismatch")
                
            h_in = input.shape[-2]
            w_in = input.shape[-1]
            output_size = [int(h_in * scale_h), int(w_in * scale_w)]

    # 3. 处理 Out-of-place 或 In-place 调用
    if out is not None:
        if not out.is_contiguous():
            raise RuntimeError("out tensor must be contiguous")
            
        _infinicore.upsample_nearest_(
            out._underlying,
            input._underlying
        )
        return out

    return Tensor(
        _infinicore.upsample_nearest(
            input._underlying,
            output_size
        )
    )

def upsample_bilinear(
    input: Tensor,
    size: Optional[Union[int, Sequence[int]]] = None,
    scale_factor: Optional[Union[float, Sequence[float]]] = None,
    align_corners: bool = False,
    *,
    out: Optional[Tensor] = None
) -> Tensor:
    r"""
    Applies bilinear interpolation upsampling to the input tensor.
    """
    if not input.is_contiguous():
        input = input.contiguous()

    if (size is None) == (scale_factor is None):
        raise ValueError("Either size or scale_factor should be defined, but not both.")

    ndim = len(input.shape)
    output_size = []
    
    if size is not None:
        if isinstance(size, int):
            if ndim == 3:
                output_size = [size]
            else:
                output_size = [size, size]
        elif isinstance(size, (list, tuple)):
            output_size = [int(s) for s in size]
        else:
            raise ValueError("size must be int or sequence of int")
    else:
        # 修正: 允许 int 和 float 类型
        scales = []
        if isinstance(scale_factor, (float, int)):
            scales = [float(scale_factor)]
        elif isinstance(scale_factor, (list, tuple)):
            scales = [float(s) for s in scale_factor]
        else:
            raise ValueError("scale_factor must be float or sequence of float")
        
        if ndim == 3:
            w_in = input.shape[-1]
            scale_w = scales[0] if len(scales) == 1 else scales[-1]
            output_size = [int(w_in * scale_w)]
        else:
            if len(scales) == 1:
                scale_h = scales[0]
                scale_w = scales[0]
            elif len(scales) >= 2:
                scale_h = scales[0]
                scale_w = scales[1]
            else:
                raise ValueError("scale_factor sequence length mismatch")
                
            h_in = input.shape[-2]
            w_in = input.shape[-1]
            output_size = [int(h_in * scale_h), int(w_in * scale_w)]

    if out is not None:
        if not out.is_contiguous():
            raise RuntimeError("out tensor must be contiguous")
            
        _infinicore.upsample_bilinear_(
            out._underlying,
            input._underlying,
            align_corners
        )
        return out

    return Tensor(
        _infinicore.upsample_bilinear(
            input._underlying,
            output_size,
            align_corners
        )
    )

def interpolate(
    input: Tensor,
    size: Optional[Union[int, Sequence[int]]] = None,
    scale_factor: Optional[Union[float, Sequence[float]]] = None,
    mode: str = 'nearest',
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None
) -> Tensor:
    r"""
    Down/up samples the input to either the given :attr:`size` or the given
    :attr:`scale_factor`

    Args:
        input (Tensor): the input tensor
        size (int or Tuple[int] or Tuple[int, int]): output spatial size.
        scale_factor (float or Tuple[float]): multiplier for spatial size.
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input and output as squares rather than points.
    """
    
    if mode == 'nearest':
        if align_corners is not None:
            raise ValueError("align_corners option can only be set with the "
                             "interpolating modes: linear | bilinear | bicubic | trilinear")
        return upsample_nearest(input, size, scale_factor)

    if mode == 'bilinear':
        if align_corners is None:
            align_corners = False
        return upsample_bilinear(input, size, scale_factor, align_corners)

    raise NotImplementedError(f"Interpolation mode '{mode}' is not currently supported.")