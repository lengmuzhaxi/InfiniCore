#ifndef __ADAPTIVE_AVG_POOL1D_CUDA_H__
#define __ADAPTIVE_AVG_POOL1D_CUDA_H__

#include <cmath>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace op::adaptive_avg_pool1d::cuda {

// ==================================================================
// 辅助函数：类型转换
// ==================================================================

// 1. 转为 float 进行累加
template <typename T>
__device__ __forceinline__ float to_float_acc(const T &x) {
    if constexpr (std::is_same_v<T, half>) {
        return __half2float(x);
    } 
    // 【修改点】直接使用 nv_bfloat16
    else if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return __bfloat162float(x);
    } 
    else if constexpr (std::is_same_v<T, double>) {
        return static_cast<float>(x); 
    } 
    else {
        return static_cast<float>(x);
    }
}

// 2. 将 float 结果转回目标类型
template <typename T>
__device__ __forceinline__ T from_float_res(float x) {
    if constexpr (std::is_same_v<T, half>) {
        return __float2half(x);
    } 
    // 【修改点】直接使用 nv_bfloat16
    else if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return __float2bfloat16(x);
    } 
    else {
        return static_cast<T>(x);
    }
}

// ==================================================================
// 核心 Kernel 实现
// ==================================================================
template <typename T>
__global__ void adaptive_avg_pool1d_kernel(
    T *output,
    const T *input,
    size_t total_elements, 
    size_t isize,          
    size_t osize           
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    size_t flat_channel_idx = idx / osize;
    size_t out_idx = idx % osize;

    const T *input_ptr = input + flat_channel_idx * isize;

    int istart = ::floorf(static_cast<float>(out_idx * isize) / osize);
    int iend   = ::ceilf(static_cast<float>((out_idx + 1) * isize) / osize);

    istart = max(0, min(istart, (int)isize));
    iend   = max(0, min(iend,   (int)isize));
    
    int klen = iend - istart;

    float sum = 0.0f;
    for (int k = istart; k < iend; ++k) {
        sum += to_float_acc(input_ptr[k]);
    }

    if (klen > 0) {
        output[idx] = from_float_res<T>(sum / klen);
    } else {
        output[idx] = from_float_res<T>(0.0f);
    }
}

} // namespace op::adaptive_avg_pool1d::cuda

#endif // __ADAPTIVE_AVG_POOL1D_CUDA_H__