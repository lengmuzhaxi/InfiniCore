#ifndef MARGIN_RANKING_LOSS_MOORE_KERNEL_H
#define MARGIN_RANKING_LOSS_MOORE_KERNEL_H

#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>
#include <cmath>
#include <cstdio>

namespace op::margin_ranking_loss::moore {

// ==================================================================
// 辅助函数: Warp & Block Reduction (用于 Mean/Sum 模式)
// ==================================================================
__device__ __forceinline__ float warpReduceSum(float val) {
    unsigned int mask = 0xffffffff;
    // MUSA 的 warpSize 通常也是 32
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(mask, val, offset);
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; 
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // 假设 Block Dim 最大 1024，即最多 32 个 Warps
    val = (threadIdx.x < (blockDim.x / 32)) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

// ==================================================================
// 辅助函数: 广播坐标映射 (根据线性索引计算物理偏移)
// ==================================================================
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
// 类型转换辅助 (适配 MUSA 半精度)
// ==================================================================
template <typename T>
__device__ __forceinline__ float cast_to_float(T val) {
    return static_cast<float>(val);
}

// 特化：针对 __maca_half 和 __maca_bfloat16
// 注意：musa_fp16.h 通常重载了 float() 转换符，如果编译报错可显式使用 __half2float
// 这里假设使用了标准的 static_cast 行为

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
    const int64_t * __restrict__ shape,    
    const int64_t * __restrict__ str1,     
    const int64_t * __restrict__ str2,     
    const int64_t * __restrict__ str_tar,  
    MarginRankingLossFunctor functor) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numel) {
        // 1. 计算偏移
        int64_t off1 = get_element_offset(idx, ndim, shape, str1);
        int64_t off2 = get_element_offset(idx, ndim, shape, str2);
        int64_t off_t = get_element_offset(idx, ndim, shape, str_tar);

        // 2. 读取并转为 float 计算
        float x1 = cast_to_float(input1[off1]);
        float x2 = cast_to_float(input2[off2]);
        float y = cast_to_float(target[off_t]);

        float loss = functor.compute(x1, x2, y);

        // 3. 写回
        output[idx] = static_cast<T>(loss);
    }
}

// ==================================================================
// Kernel 2: Reduction (Mean / Sum) - 使用 Grid-Stride Loop
// ==================================================================
template <typename T>
__global__ void margin_ranking_loss_reduce_kernel(
    float * output,                        // [1] Accumulator
    const T * __restrict__ input1,
    const T * __restrict__ input2,
    const T * __restrict__ target,
    size_t numel,
    int ndim,
    const int64_t * __restrict__ shape,    
    const int64_t * __restrict__ str1,     
    const int64_t * __restrict__ str2,     
    const int64_t * __restrict__ str_tar,  
    MarginRankingLossFunctor functor,
    float scale                            // Mean: 1/N, Sum: 1.0
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;

    for (size_t i = idx; i < numel; i += stride) {
        int64_t off1 = get_element_offset(i, ndim, shape, str1);
        int64_t off2 = get_element_offset(i, ndim, shape, str2);
        int64_t off_t = get_element_offset(i, ndim, shape, str_tar);

        float x1 = cast_to_float(input1[off1]);
        float x2 = cast_to_float(input2[off2]);
        float y = cast_to_float(target[off_t]);

        local_sum += functor.compute(x1, x2, y);
    }

    // Block 内归约
    float block_sum = blockReduceSum(local_sum);

    // Global Atomic Add
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum * scale);
    }
}

// ==================================================================
// Kernel 3: 将 Float 累加结果转回 T
// ==================================================================
template <typename T>
__global__ void cast_float_to_t(T* output, const float* src) {
    if (threadIdx.x == 0) {
        *output = static_cast<T>(*src);
    }
}

} // namespace op::margin_ranking_loss::moore

#endif // MARGIN_RANKING_LOSS_MOORE_KERNEL_H