#include "adaptive_avg_pool1d_nvidia.cuh"
#include "../cuda/kernel.cuh"
#include "../../../handle.h" 

namespace op::adaptive_avg_pool1d::nvidia {

// ==================================================================
// Kernel Launch Helper
// ==================================================================
template <typename T>
void launch_kernel(
    void *output, 
    const void *input, 
    size_t num_channels, 
    size_t isize, 
    size_t osize, 
    void *stream) {
    
    auto out_ptr = reinterpret_cast<T *>(output);
    auto in_ptr = reinterpret_cast<const T *>(input);
    
    // 计算总输出元素个数: N * C * L_out
    size_t total_elements = num_channels * osize;
    
    // CUDA Grid/Block 配置
    size_t block_size = 256;
    size_t grid_size = (total_elements + block_size - 1) / block_size;

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // 调用 cuda 命名空间下的 kernel
    cuda::adaptive_avg_pool1d_kernel<T><<<grid_size, block_size, 0, cuda_stream>>>(
        out_ptr,
        in_ptr,
        total_elements,
        isize,
        osize
    );
}

// ==================================================================
// Descriptor Implementation
// ==================================================================

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc) {

    // 1. 使用 Info 类解析参数
    auto info_result = AdaptiveAvgPool1dInfo::create(out_desc, in_desc);
    if (!info_result) {
        return info_result.status();
    }
    auto info = info_result.take();

    // 2. 创建 Descriptor
    *desc_ptr = new Descriptor(
        new Opaque(),        // Opaque 指针
        info,                // Info 对象
        0,                   // Workspace size
        handle->device,      // Device Type
        handle->device_id    // Device ID
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
    auto num_channels = _info.num_channels();
    auto input_size = _info.input_size();
    auto output_size = _info.output_size();

    switch (dtype) {
    case INFINI_DTYPE_F16:
        launch_kernel<half>(output, input, num_channels, input_size, output_size, stream);
        break;
    case INFINI_DTYPE_BF16:
        // 【修改 2】使用标准类型 nv_bfloat16 替代 cuda_bfloat16
        launch_kernel<nv_bfloat16>(output, input, num_channels, input_size, output_size, stream);
        break;
    case INFINI_DTYPE_F32:
        launch_kernel<float>(output, input, num_channels, input_size, output_size, stream);
        break;
    case INFINI_DTYPE_F64:
        launch_kernel<double>(output, input, num_channels, input_size, output_size, stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::adaptive_avg_pool1d::nvidia