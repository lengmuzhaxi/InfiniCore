#include "pixel_shuffle_cpu.h"
#include <cstdint>
#include <stddef.h>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "../../../devices/cpu/common_cpu.h"
#include "../../../handle.h"

namespace op::pixel_shuffle::cpu {

Descriptor::~Descriptor() = default;

// ==================================================================
// 辅助函数：通用 stride 寻址
// ==================================================================

inline size_t offset_4d(size_t n, size_t c, size_t h, size_t w, const int64_t *strides) {
    return n * strides[0] + c * strides[1] + h * strides[2] + w * strides[3];
}

// ==================================================================
// 核心 Kernel 实现
// ==================================================================

template <typename Tdata>
void calculate_impl(
    const PixelShuffleInfo &info,
    void *output,
    const void *input) {

    int64_t upscale_factor = info.upscale_factor();
    
    // 获取维度
    size_t batch = info.n();
    size_t c_out = info.c();
    size_t h_out = info.h();
    size_t w_out = info.w();

    Tdata *out_ptr = reinterpret_cast<Tdata *>(output);
    const Tdata *inp_ptr = reinterpret_cast<const Tdata *>(input);

    const int64_t *out_strides = info.out_strides().data();
    const int64_t *in_strides = info.in_strides().data();

    for (size_t n = 0; n < batch; ++n) {
        for (size_t c = 0; c < c_out; ++c) {
            for (size_t h = 0; h < h_out; ++h) {
                for (size_t w = 0; w < w_out; ++w) {
                    
                    // 映射回 Input 坐标
                    size_t h_in = h / upscale_factor;
                    size_t w_in = w / upscale_factor;
                    
                    size_t offset_h = h % upscale_factor;
                    size_t offset_w = w % upscale_factor;
                    
                    size_t c_in = c * (upscale_factor * upscale_factor) + 
                                  offset_h * upscale_factor + 
                                  offset_w;

                    size_t out_idx = offset_4d(n, c, h, w, out_strides);
                    size_t in_idx = offset_4d(n, c_in, h_in, w_in, in_strides);

                    out_ptr[out_idx] = inp_ptr[in_idx];
                }
            }
        }
    }
}

// ==================================================================
// Descriptor 接口实现
// ==================================================================

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc,
    int64_t upscale_factor) {

    auto dtype = out_desc->dtype();
    // 仅保留浮点类型检查
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16, INFINI_DTYPE_F64);
    
    auto result = PixelShuffleInfo::create(out_desc, in_desc, upscale_factor);
    CHECK_RESULT(result);

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    *desc_ptr = new Descriptor(
        nullptr,
        result.take(),
        0, 
        handle->device,
        handle->device_id
    );

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {

    auto dtype = _info.dtype();

    switch (dtype) {
    case INFINI_DTYPE_F16:
        calculate_impl<fp16_t>(_info, output, input);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_BF16:
        calculate_impl<bf16_t>(_info, output, input);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F32:
        calculate_impl<float>(_info, output, input);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F64:
        calculate_impl<double>(_info, output, input);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::pixel_shuffle::cpu