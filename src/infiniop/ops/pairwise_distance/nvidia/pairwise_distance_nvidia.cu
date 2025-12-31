#include "pairwise_distance_nvidia.cuh"
#include "../cuda/kernel.cuh"
#include "../../../handle.h"
#include <cstdint>
#include <algorithm>

namespace op::pairwise_distance::nvidia {

template <typename T>
static inline bool is_aligned(const void *ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

// ==================================================================
// Kernel Launch Logic
// ==================================================================
template <typename T>
void launch_kernel(
    void *output, 
    const void *x1, 
    const void *x2, 
    const PairwiseDistanceInfo& info,
    void *stream) {

    // 1. 准备指针
    auto out_ptr = reinterpret_cast<T *>(output);
    auto x1_ptr = reinterpret_cast<const T *>(x1);
    auto x2_ptr = reinterpret_cast<const T *>(x2);
    
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    
    // 2. 准备参数
    size_t N = info.batch_size();
    size_t D = info.feature_dim();
    
    // 创建 Functor
    op::pairwise_distance::cuda::PairwiseDistanceFunctor functor(
        info.p(), 
        info.eps()
    );

    // 3. 配置 Kernel
    // 每个线程处理一个样本 N
    size_t block_size = 256;
    size_t grid_size = (N + block_size - 1) / block_size;
    
    op::pairwise_distance::cuda::pairwise_distance_kernel<T>
        <<<grid_size, block_size, 0, cuda_stream>>>(
            out_ptr, x1_ptr, x2_ptr, N, D, functor
        );
}

// ==================================================================
// Descriptor 实现
// ==================================================================
struct Descriptor::Opaque {};

Descriptor::~Descriptor() { 
    if (_opaque) delete _opaque; 
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc, 
    infiniopTensorDescriptor_t x1_desc, 
    infiniopTensorDescriptor_t x2_desc, 
    float p, 
    float eps, 
    bool keepdim) {

    // 创建 Info
    auto info_result = PairwiseDistanceInfo::create(out_desc, x1_desc, x2_desc, p, eps, keepdim);
    if (!info_result) return info_result.status();
    
    // PairwiseDistance 计算通常在寄存器内完成，不需要额外的 Workspace
    size_t workspace_size = 0;

    *desc_ptr = new Descriptor(new Opaque(), info_result.take(), workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, 
    size_t workspace_size, 
    void *output,
    const void *x1, 
    const void *x2, 
    void *stream) const {

    auto dtype = _info.dtype();

    // 简单的 workspace 检查 (虽然这里 size 为 0)
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    switch (dtype) {
    case INFINI_DTYPE_F16:
        launch_kernel<half>(output, x1, x2, _info, stream);
        break;
    case INFINI_DTYPE_BF16:
        launch_kernel<nv_bfloat16>(output, x1, x2, _info, stream);
        break;
    case INFINI_DTYPE_F32:
        launch_kernel<float>(output, x1, x2, _info, stream);
        break;
    case INFINI_DTYPE_F64:
        launch_kernel<double>(output, x1, x2, _info, stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::pairwise_distance::nvidia