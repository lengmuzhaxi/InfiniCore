#ifndef __LOG_SOFTMAX_CUDA_CUH__
#define __LOG_SOFTMAX_CUDA_CUH__

#include <cuda_runtime.h>
#if defined ENABLE_METAX_API
    #include <maca_fp16.h>
    #include <maca_bfloat16.h>
#else
    #include <cuda_fp16.h>
    #include <cuda_bf16.h>
#endif

#include <cmath>
#include <limits>
#include <cstdint>

namespace op::log_softmax::cuda {

// ==================================================================
// 类型转换辅助 (保证中间计算精度为 float)
// ==================================================================
template <typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

// ==================================================================
// Warp Reduction Helpers
// ==================================================================
template <typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        #if defined ENABLE_METAX_API
            // 适配 MetaX API 如果需要
            val = max(val, __shfl_down_sync(0xffffffff, val, offset));
        #else
            val = max(val, __shfl_down_sync(0xffffffff, val, offset));
        #endif
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ==================================================================
// Block Reduction Helpers
// ==================================================================
template <typename T>
__device__ __forceinline__ T block_reduce_max(T val) {
    static __shared__ float shared[32]; // Max 32 warps per block
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_max(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // 假设 BlockDim.x 不超过 1024 (32 warps)
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -INFINITY;
    
    if (wid == 0) val = warp_reduce_max(val);
    
    return val;
}

template <typename T>
__device__ __forceinline__ T block_reduce_sum(T val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}

// ==================================================================
// Kernel: LogSoftmax (Online Softmax / 3-Pass Algorithm)
// ==================================================================
// 假设:
// 1. Grid 处理 Outer * Inner 个 Slice (Rows)
// 2. Block 处理 1 个 Slice (沿着 Dim 维度)
// 3. 使用 Loop 处理 Dim > BlockDim 的情况
template <typename T>
__global__ void log_softmax_kernel(
    T * __restrict__ output,        // [Outer, Dim, Inner]
    const T * __restrict__ input,   // [Outer, Dim, Inner]
    size_t dim_size,
    size_t inner_size
) {
    // 共享内存用于存储 Block Reduction 的结果广播
    __shared__ float s_max;
    __shared__ float s_sum;

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;

    // 1. 计算当前 Slice 的基地址
    // GridDim.x = Outer * Inner
    size_t outer_idx = bid / inner_size;
    size_t inner_idx = bid % inner_size;

    // Layout: [outer, dim, inner]
    // Base offset = outer * (dim_size * inner_size) + inner_idx
    size_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
    size_t stride = inner_size; // 元素在 Dim 维度的跨度

    // ============================================================
    // Pass 1: Find Max (为了数值稳定性)
    // ============================================================
    float local_max = -INFINITY;
    for (size_t i = tid; i < dim_size; i += blockDim.x) {
        float val = to_float(input[base_offset + i * stride]);
        if (val > local_max) {
            local_max = val;
        }
    }
    
    // Block Reduction 得到全局 Max
    float global_max = block_reduce_max(local_max);
    if (tid == 0) s_max = global_max;
    __syncthreads();
    global_max = s_max; // 广播

    // ============================================================
    // Pass 2: Calculate Sum of Exponentials
    // sum(exp(x - max))
    // ============================================================
    float local_sum = 0.0f;
    for (size_t i = tid; i < dim_size; i += blockDim.x) {
        float val = to_float(input[base_offset + i * stride]);
        local_sum += expf(val - global_max);
    }

    // Block Reduction 得到全局 Sum
    float global_sum = block_reduce_sum(local_sum);
    if (tid == 0) s_sum = global_sum;
    __syncthreads();
    global_sum = s_sum; // 广播

    // 计算 LogSumExp: log(sum) + max
    float log_sum_exp = logf(global_sum) + global_max;

    // ============================================================
    // Pass 3: Calculate Final Output
    // output = x - LogSumExp
    // ============================================================
    for (size_t i = tid; i < dim_size; i += blockDim.x) {
        size_t idx = base_offset + i * stride;
        float val = to_float(input[idx]);
        output[idx] = static_cast<T>(val - log_sum_exp);
    }
}

} // namespace op::log_softmax::cuda

#endif // __LOG_SOFTMAX_CUDA_CUH__