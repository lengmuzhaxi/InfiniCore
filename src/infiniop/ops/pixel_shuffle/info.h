#ifndef __PIXEL_SHUFFLE_INFO_H__
#define __PIXEL_SHUFFLE_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::pixel_shuffle {

class PixelShuffleInfo {
    PixelShuffleInfo() = default;

public:
    int _dtype;
    int64_t _upscale_factor;
    
    // 存储输出张量的维度 N, C, H, W (用于 Kernel Launch 配置)
    size_t _n;
    size_t _c;
    size_t _h;
    size_t _w;

    // 存储 Stride 信息 (用于 Kernel 中的指针计算)
    std::vector<int64_t> _out_strides;
    std::vector<int64_t> _in_strides;

    // 构造函数
    PixelShuffleInfo(int dtype, int64_t upscale_factor,
                     size_t n, size_t c, size_t h, size_t w,
                     std::vector<int64_t> out_strides,
                     std::vector<int64_t> in_strides)
        : _dtype(dtype), _upscale_factor(upscale_factor),
          _n(n), _c(c), _h(h), _w(w),
          _out_strides(std::move(out_strides)),
          _in_strides(std::move(in_strides)) {}

    // Getter 方法
    int dtype() const { return _dtype; }
    int64_t upscale_factor() const { return _upscale_factor; }
    
    size_t n() const { return _n; }
    size_t c() const { return _c; }
    size_t h() const { return _h; }
    size_t w() const { return _w; }

    const std::vector<int64_t>& out_strides() const { return _out_strides; }
    const std::vector<int64_t>& in_strides() const { return _in_strides; }

    static utils::Result<PixelShuffleInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t in_desc,
        int64_t upscale_factor) {

        // 1. 检查 upscale_factor 有效性
        if (upscale_factor <= 0) {
            return INFINI_STATUS_BAD_PARAM;
        }

        // 2. 检查数据类型一致性
        if (out_desc->dtype() != in_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        int dtype = in_desc->dtype();

        // 3. 检查维度数量 (必须是 4D: N, C, H, W)
        if (in_desc->ndim() != 4 || out_desc->ndim() != 4) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        const auto &in_shape = in_desc->shape();
        const auto &out_shape = out_desc->shape();

        // 4. 检查维度变换逻辑
        // Input:  [N, C * r^2, H, W]
        // Output: [N, C, H * r, W * r]
        
        // N 维度必须相等
        if (in_shape[0] != out_shape[0]) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        
        // C 维度: in_C 必须等于 out_C * r * r
        if (in_shape[1] != out_shape[1] * upscale_factor * upscale_factor) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // H 维度: out_H 必须等于 in_H * r
        if (out_shape[2] != in_shape[2] * upscale_factor) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // W 维度: out_W 必须等于 in_W * r
        if (out_shape[3] != in_shape[3] * upscale_factor) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // 5. 提取 Strides
        // 注意：Descriptor 中的 vector 是通过值传递或移动的
        std::vector<int64_t> out_strides = out_desc->strides();
        std::vector<int64_t> in_strides = in_desc->strides();

        // 6. 返回 Info 对象
        return utils::Result<PixelShuffleInfo>(PixelShuffleInfo{
            dtype,
            upscale_factor,
            static_cast<size_t>(out_shape[0]), // n
            static_cast<size_t>(out_shape[1]), // c (out)
            static_cast<size_t>(out_shape[2]), // h (out)
            static_cast<size_t>(out_shape[3]), // w (out)
            std::move(out_strides),
            std::move(in_strides)
        });
    }
};

} // namespace op::pixel_shuffle

#endif // __PIXEL_SHUFFLE_INFO_H__