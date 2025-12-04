#ifndef __ADDBMM_NVIDIA_CUH__
#define __ADDBMM_NVIDIA_CUH__

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>

namespace op::addbmm::nvidia {

// --- 类型转换辅助函数 ---
template <typename T>
__device__ __forceinline__ float to_float_acc(const T &x) {
    if constexpr (std::is_same_v<T, half>) return __half2float(x);
    else if constexpr (std::is_same_v<T, nv_bfloat16>) return __bfloat162float(x);
    else return static_cast<float>(x);
}

template <typename T>
__device__ __forceinline__ T from_float_res(float x) {
    if constexpr (std::is_same_v<T, half>) return __float2half(x);
    else if constexpr (std::is_same_v<T, nv_bfloat16>) return __float2bfloat16(x);
    else return static_cast<T>(x);
}

// ==================================================================
// 1. 核心 Kernel 实现 (设备端代码)
// ==================================================================
template <typename T>
__global__ void addbmm_kernel(
    T *output,
    const T *input,
    const T *batch1,
    const T *batch2,
    size_t b, size_t n, size_t m, size_t p,
    float alpha, float beta,  // 【统一顺序】alpha 在前，beta 在后
    // Strides
    ptrdiff_t out_s0, ptrdiff_t out_s1,
    ptrdiff_t inp_s0, ptrdiff_t inp_s1,
    ptrdiff_t b1_s0, ptrdiff_t b1_s1, ptrdiff_t b1_s2,
    ptrdiff_t b2_s0, ptrdiff_t b2_s1, ptrdiff_t b2_s2
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n * p) return;

    size_t row = idx / p;
    size_t col = idx % p;

    float matmul_sum = 0.0f;
    for (size_t i = 0; i < b; ++i) {
        for (size_t k = 0; k < m; ++k) {
            size_t b1_idx = i * b1_s0 + row * b1_s1 + k * b1_s2;
            size_t b2_idx = i * b2_s0 + k * b2_s1 + col * b2_s2;
            
            matmul_sum += to_float_acc(batch1[b1_idx]) * to_float_acc(batch2[b2_idx]);
        }
    }

    float res = 0.0f;
    if (beta != 0.0f) {
        size_t inp_idx = row * inp_s0 + col * inp_s1;
        // 【注意】这里使用了 alpha, beta 的顺序，如果测试失败，需要怀疑 InfiniOP 内部的约定。
        res = beta * to_float_acc(input[inp_idx]);
    }
    res += alpha * matmul_sum;

    size_t out_idx = row * out_s0 + col * out_s1;
    output[out_idx] = from_float_res<T>(res);
}


// ==================================================================
// 2. 启动 Kernel 的主机端函数 (Host Function)
// ==================================================================
// 修复 launch_kernel is undefined 错误
template <typename T>
void launch_kernel(
    void *output, const void *input, const void *batch1, const void *batch2, 
    size_t b, size_t n, size_t m, size_t p, 
    float alpha, float beta, // 接收 alpha, beta (顺序匹配上面的 kernel)
    // Strides
    ptrdiff_t out_s0, ptrdiff_t out_s1,
    ptrdiff_t inp_s0, ptrdiff_t inp_s1,
    ptrdiff_t b1_s0, ptrdiff_t b1_s1, ptrdiff_t b1_s2,
    ptrdiff_t b2_s0, ptrdiff_t b2_s1, ptrdiff_t b2_s2,
    void *stream) {

    auto out_ptr = reinterpret_cast<T *>(output);
    auto in_ptr = reinterpret_cast<const T *>(input);
    auto b1_ptr = reinterpret_cast<const T *>(batch1);
    auto b2_ptr = reinterpret_cast<const T *>(batch2);

    // 配置 1D Grid (与 Kernel 保持一致)
    size_t total_elements = n * p;
    size_t block_size = 256;
    size_t grid_size = (total_elements + block_size - 1) / block_size;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // 启动 Kernel
    addbmm_kernel<T><<<grid_size, block_size, 0, cuda_stream>>>(
        out_ptr, in_ptr, b1_ptr, b2_ptr,
        b, n, m, p,
        alpha, beta, // 传参顺序匹配 Kernel 定义
        out_s0, out_s1, inp_s0, inp_s1, 
        b1_s0, b1_s1, b1_s2, b2_s0, b2_s1, b2_s2
    );
}


} // namespace op::addbmm::nvidia

#endif // __ADDBMM_NVIDIA_CUH__