#ifndef __SPLIT_CUDA_KERNEL_CUH__
#define __SPLIT_CUDA_KERNEL_CUH__

#include <cuda_runtime.h>
#include <type_traits>

#if defined ENABLE_METAX_API
    #include <maca_fp16.h>
    #include <maca_bfloat16.h>
#else
    #include <cuda_fp16.h>
    #include <cuda_bf16.h>
#endif

namespace op::split::cuda {

// 最大支持维度 (需与 Info 定义保持一致)
constexpr int MAX_NDIM = 8;

// 用于传给 Kernel 的轻量级元数据结构
struct TensorMeta {
    size_t shape[MAX_NDIM];
    ptrdiff_t strides[MAX_NDIM];
    int ndim;
};

template <typename T>
__global__ void split_kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    TensorMeta out_meta,
    TensorMeta in_meta,
    int64_t axis,
    size_t axis_offset,
    size_t total_elements) {

    // 计算当前线程负责的线性索引 (相对于当前 Output Tensor)
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        size_t remaining = idx;
        size_t in_phys_offset = 0;
        size_t out_phys_offset = 0;

        // 将线性索引转换为多维坐标，并计算物理偏移
        // 从最高维倒序计算
        #pragma unroll
        for (int d = MAX_NDIM - 1; d >= 0; --d) {
            if (d < out_meta.ndim) {
                size_t dim_sz = out_meta.shape[d];
                size_t dim_coord = remaining % dim_sz;
                remaining /= dim_sz;

                // 1. 计算 Output 物理偏移
                out_phys_offset += dim_coord * out_meta.strides[d];
                size_t in_coord = (d == axis) ? (dim_coord + axis_offset) : dim_coord;
                in_phys_offset += in_coord * in_meta.strides[d];
            }
        }
        output[out_phys_offset] = input[in_phys_offset];
    }
}

} // namespace op::split::cuda

#endif // __SPLIT_CUDA_KERNEL_CUH__