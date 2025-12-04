#ifndef __AFFINE_GRID_CUDA_H__
#define __AFFINE_GRID_CUDA_H__

#include <cmath>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace op::affine_grid::cuda {

// ==================================================================
// 辅助函数：类型转换 (与 AdaptiveAvgPool1d 保持一致)
// ==================================================================

// 1. 转为 float 进行高精度计算
template <typename T>
__device__ __forceinline__ float to_float_acc(const T &x) {
    if constexpr (std::is_same_v<T, half>) {
        return __half2float(x);
    } 
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
__global__ void affine_grid_kernel(
    T *output,        // Output Shape: (N, H, W, 2)
    const T *theta,   // Input Shape:  (N, 2, 3)
    size_t N,         // Batch Size
    size_t H,         // Output Height
    size_t W,         // Output Width
    bool align_corners
) {
    // 每个线程处理一个输出像素点 (n, h, w)
    // 也就是生成一个 (x, y) 坐标对
    size_t total_elements = N * H * W;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    // 1. 解构索引
    // idx = n * (H * W) + h * W + w
    size_t w = idx % W;
    size_t h = (idx / W) % H;
    size_t n = idx / (H * W);

    // 2. 计算归一化基准坐标 (Base Coordinates)
    // 范围通常是 [-1, 1]
    float x_norm, y_norm;

    if (align_corners) {
        // align_corners = True: -1 指向像素中心
        // 公式: 2 * i / (size - 1) - 1
        x_norm = (W > 1) ? (2.0f * w) / (W - 1.0f) - 1.0f : 0.0f;
        y_norm = (H > 1) ? (2.0f * h) / (H - 1.0f) - 1.0f : 0.0f;
    } else {
        // align_corners = False: -1 指向像素角点
        // 公式: (2 * i + 1) / size - 1
        x_norm = (2.0f * w + 1.0f) / W - 1.0f;
        y_norm = (2.0f * h + 1.0f) / H - 1.0f;
    }

    // 3. 读取仿射矩阵 Theta
    // Theta shape is (N, 2, 3), flattened stride is 6
    const T* theta_ptr = theta + n * 6;
    
    // 加载参数并转为 float
    float r00 = to_float_acc(theta_ptr[0]); // r11
    float r01 = to_float_acc(theta_ptr[1]); // r12
    float tx  = to_float_acc(theta_ptr[2]); // tx
    float r10 = to_float_acc(theta_ptr[3]); // r21
    float r11 = to_float_acc(theta_ptr[4]); // r22
    float ty  = to_float_acc(theta_ptr[5]); // ty

    // 4. 应用仿射变换 (Matrix Multiplication)
    // [ x_grid ]   [ r00 r01 tx ]   [ x_norm ]
    // [ y_grid ] = [ r10 r11 ty ] * [ y_norm ]
    //                               [   1    ]
    float grid_x = r00 * x_norm + r01 * y_norm + tx;
    float grid_y = r10 * x_norm + r11 * y_norm + ty;

    // 5. 写回输出
    // Output shape is (N, H, W, 2). 
    // 内存布局: [n, h, w, 0] 是 x, [n, h, w, 1] 是 y
    // idx 对应的是 (n, h, w) 这一组，所以偏移量是 idx * 2
    output[idx * 2 + 0] = from_float_res<T>(grid_x);
    output[idx * 2 + 1] = from_float_res<T>(grid_y);
}

} // namespace op::affine_grid::cuda

#endif // __AFFINE_GRID_CUDA_H__