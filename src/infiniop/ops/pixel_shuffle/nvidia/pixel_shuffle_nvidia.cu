#include "pixel_shuffle_nvidia.cuh"
#include "../../../handle.h"
#include "../cuda/kernel.cuh"
#include "../../../devices/nvidia/nvidia_handle.h"   
namespace op::pixel_shuffle::nvidia {

// ==================================================================
// Descriptor Implementation
// ==================================================================

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

// 创建算子描述符
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc,
    int64_t upscale_factor) {

    // 1. 使用 Info 类解析并校验参数
    auto info_result = PixelShuffleInfo::create(out_desc, in_desc, upscale_factor);
    if (!info_result) {
        return info_result.status();
    }
    auto info = info_result.take();

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);

    // 2. 创建 Descriptor
    *desc_ptr = new Descriptor(
        new Opaque(),           // Opaque 指针
        info,                   // Info 对象
        0,                      // Workspace size (PixelShuffle 不需要 workspace)
        handle->device,         // Device Type
        handle->device_id       // Device ID
    );

    return INFINI_STATUS_SUCCESS;
}

// 执行计算
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {

    // 从 _info 中提取必要的维度参数
    auto dtype = _info.dtype();
    
    size_t batch = _info.n();
    size_t c_out = _info.c();
    size_t h_out = _info.h();
    size_t w_out = _info.w();
    int64_t upscale_factor = _info.upscale_factor();

    const int64_t *out_strides = _info.out_strides().data();
    const int64_t *in_strides = _info.in_strides().data();

    // 根据数据类型分发
    switch (dtype) {
    case INFINI_DTYPE_F16:
        launch_kernel<half>(
            output, input, 
            batch, c_out, h_out, w_out, upscale_factor,
            out_strides[0], out_strides[1], out_strides[2], out_strides[3],
            in_strides[0], in_strides[1], in_strides[2], in_strides[3],
            stream);
        break;

    case INFINI_DTYPE_BF16:
        launch_kernel<nv_bfloat16>(
            output, input, 
            batch, c_out, h_out, w_out, upscale_factor,
            out_strides[0], out_strides[1], out_strides[2], out_strides[3],
            in_strides[0], in_strides[1], in_strides[2], in_strides[3],
            stream);
        break;

    case INFINI_DTYPE_F32:
        launch_kernel<float>(
            output, input, 
            batch, c_out, h_out, w_out, upscale_factor,
            out_strides[0], out_strides[1], out_strides[2], out_strides[3],
            in_strides[0], in_strides[1], in_strides[2], in_strides[3],
            stream);
        break;

    case INFINI_DTYPE_F64:
        launch_kernel<double>(
            output, input, 
            batch, c_out, h_out, w_out, upscale_factor,
            out_strides[0], out_strides[1], out_strides[2], out_strides[3],
            in_strides[0], in_strides[1], in_strides[2], in_strides[3],
            stream);
        break;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::pixel_shuffle::nvidia