#ifndef __UPSAMPLE_NEAREST_CUDA_CUH__
#define __UPSAMPLE_NEAREST_CUDA_CUH__

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

namespace op::upsample_nearest::cuda {

// ==================================================================
// 辅助函数: 计算最近邻源坐标索引
// ==================================================================
__device__ __forceinline__ int get_nearest_index(
    int out_index,
    float scale,
    int input_size) {
    
    // Nearest neighbor logic: floor(out_index * scale)
    // 使用 floorf 确保向下取整
    int idx = static_cast<int>(floorf(out_index * scale));
    
    // 边界处理 (Clamping): 限制在 [0, input_size - 1] 范围内
    // 虽然理论上 idx < input_size，但为了防止浮点精度误差导致越界，加上 min 保护
    return min(max(idx, 0), input_size - 1);
}

// ==================================================================
// Kernel: 最近邻插值核心逻辑
// ==================================================================
template <typename T>
__global__ void upsample_nearest_kernel(
    T * __restrict__ output,        // [N, C, H_out, W_out]
    const T * __restrict__ input,   // [N, C, H_in, W_in]
    size_t N,
    size_t C,
    size_t H_in,
    size_t W_in,
    size_t H_out,
    size_t W_out,
    float scale_h,                  // 预计算的缩放比例 (in_size / out_size)
    float scale_w) {                // 预计算的缩放比例 (in_size / out_size)

    // Grid-Stride Loop: 处理每一个输出元素
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = N * C * H_out * W_out;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < total_elements; i += stride) {
        // 1. 解构索引 (N, C, H_out, W_out)
        // Layout: NCHW
        size_t w_out_idx = i % W_out;
        size_t temp = i / W_out;
        size_t h_out_idx = temp % H_out;
        temp /= H_out;
        size_t c_idx = temp % C;
        size_t n_idx = temp / C;

        // 2. 计算源索引 (Source Indices)
        int h_in_idx = get_nearest_index(static_cast<int>(h_out_idx), scale_h, static_cast<int>(H_in));
        int w_in_idx = get_nearest_index(static_cast<int>(w_out_idx), scale_w, static_cast<int>(W_in));

        // 3. 计算输入数据的线性偏移量
        // Input layout: [N, C, H_in, W_in]
        size_t in_offset = (n_idx * C + c_idx) * H_in * W_in + h_in_idx * W_in + w_in_idx;

        // 4. 读取并写入数据 (直接赋值，无插值)
        output[i] = input[in_offset];
    }
}

} // namespace op::upsample_nearest::cuda

#endif // __UPSAMPLE_NEAREST_CUDA_CUH__