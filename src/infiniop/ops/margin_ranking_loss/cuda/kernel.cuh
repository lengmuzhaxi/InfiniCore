#ifndef __MARGIN_RANKING_LOSS_CUDA_CUH__
#define __MARGIN_RANKING_LOSS_CUDA_CUH__

#include <cuda_runtime.h>
#if defined ENABLE_METAX_API
    #include <maca_fp16.h>
    #include <maca_bfloat16.h>
    using nv_bfloat162 = __maca_bfloat162;
#else
    #include <cuda_fp16.h>
    #include <cuda_bf16.h>
#endif

#include <cmath>
#include <cstdio>

namespace op::margin_ranking_loss::cuda {

// ==================================================================
// 辅助函数: 归约 (Warp & Block Reduction)
// ==================================================================
__device__ __forceinline__ float warpReduceSum(float val) {
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(mask, val, offset);
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; // Max 1024 threads / 32 warps
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

// ==================================================================
// 辅助函数: 广播坐标映射
// ==================================================================
// 根据输出的线性索引，结合形状和广播步长，计算输入 Tensor 的物理偏移量
__device__ __forceinline__ int64_t get_element_offset(
    size_t linear_idx,
    int ndim,
    const int64_t* __restrict__ shape,   // Output Shape
    const int64_t* __restrict__ strides) // Input Effective Strides
{
    int64_t offset = 0;
    size_t remainder = linear_idx;

    #pragma unroll
    for (int i = ndim - 1; i >= 0; --i) {
        int64_t dim_size = shape[i];
        int64_t coord = remainder % dim_size;
        remainder /= dim_size;
        offset += coord * strides[i];
    }
    return offset;
}

// ==================================================================
// Functor: 核心数学逻辑
// ==================================================================
struct MarginRankingLossFunctor {
    float margin;

    __host__ __device__ MarginRankingLossFunctor(float m) : margin(m) {}

    // loss(x1, x2, y) = max(0, -y * (x1 - x2) + margin)
    __device__ __forceinline__ float compute(float x1, float x2, float y) const {
        float val = -y * (x1 - x2) + margin;
        return (val > 0.0f) ? val : 0.0f;
    }
};

// ==================================================================
// Kernel 1: Element-wise (Reduction = None)
// ==================================================================
template <typename T>
__global__ void margin_ranking_loss_kernel(
    T * __restrict__ output,
    const T * __restrict__ input1,
    const T * __restrict__ input2,
    const T * __restrict__ target,
    size_t numel,
    int ndim,
    const int64_t * __restrict__ shape,    // Output Shape
    const int64_t * __restrict__ str1,     // Strides for Input1
    const int64_t * __restrict__ str2,     // Strides for Input2
    const int64_t * __restrict__ str_tar,  // Strides for Target
    MarginRankingLossFunctor functor) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numel) {
        // 1. 计算各输入的偏移量
        int64_t off1 = get_element_offset(idx, ndim, shape, str1);
        int64_t off2 = get_element_offset(idx, ndim, shape, str2);
        int64_t off_t = get_element_offset(idx, ndim, shape, str_tar);

        // 2. 读取数据并转换精度
        float x1 = static_cast<float>(input1[off1]);
        float x2 = static_cast<float>(input2[off2]);
        float y = static_cast<float>(target[off_t]);

        // 3. 计算 Loss
        float loss = functor.compute(x1, x2, y);

        // 4. 写入输出
        output[idx] = static_cast<T>(loss);
    }
}

// ==================================================================
// Kernel 2: Reduction (Mean / Sum)
// ==================================================================
template <typename T>
__global__ void margin_ranking_loss_reduce_kernel(
    float * output,                        // [1] Accumulator (Float)
    const T * __restrict__ input1,
    const T * __restrict__ input2,
    const T * __restrict__ target,
    size_t numel,
    int ndim,
    const int64_t * __restrict__ shape,    // Output Shape
    const int64_t * __restrict__ str1,     // Strides for Input1
    const int64_t * __restrict__ str2,     // Strides for Input2
    const int64_t * __restrict__ str_tar,  // Strides for Target
    MarginRankingLossFunctor functor,
    float scale                            // Mean: 1/N, Sum: 1.0
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;

    // Grid-Stride Loop
    for (size_t i = idx; i < numel; i += stride) {
        // 1. 计算偏移量
        int64_t off1 = get_element_offset(i, ndim, shape, str1);
        int64_t off2 = get_element_offset(i, ndim, shape, str2);
        int64_t off_t = get_element_offset(i, ndim, shape, str_tar);

        // 2. 读取数据
        float x1 = static_cast<float>(input1[off1]);
        float x2 = static_cast<float>(input2[off2]);
        float y = static_cast<float>(target[off_t]);

        // 3. 累加 Loss
        local_sum += functor.compute(x1, x2, y);
    }

    // Block Reduction
    float block_sum = blockReduceSum(local_sum);

    // Global Atomic Add
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum * scale);
    }
}

// ==================================================================
// Kernel 3: Cast Float Accumulator back to T
// ==================================================================
template <typename T>
__global__ void cast_float_to_t(T* output, const float* src) {
    *output = static_cast<T>(*src);
}

} // namespace op::margin_ranking_loss::cuda

#endif // __MARGIN_RANKING_LOSS_CUDA_CUH__