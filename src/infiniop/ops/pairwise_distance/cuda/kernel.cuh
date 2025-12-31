#ifndef __PAIRWISE_DISTANCE_CUDA_CUH__
#define __PAIRWISE_DISTANCE_CUDA_CUH__

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

namespace op::pairwise_distance::cuda {

// ==================================================================
// Functor: 核心数学逻辑
// ==================================================================
struct PairwiseDistanceFunctor {
    float p;
    float eps;

    __host__ __device__ PairwiseDistanceFunctor(float p_, float eps_) 
        : p(p_), eps(eps_) {}

    // 辅助函数: 计算两个向量 x, y 之间的 p-范数距离
    // x, y 指针，长度 D
    template <typename T>
    __device__ __forceinline__ float compute_dist(const T* x, const T* y, size_t D) const {
        float sum = 0.0f;
        
        // p = 1.0: Manhattan Distance
        if (abs(p - 1.0f) < 1e-6f) {
            for (size_t i = 0; i < D; ++i) {
                float diff = fabsf(static_cast<float>(x[i]) - static_cast<float>(y[i]));
                sum += diff;
            }
            return sum + eps;
        }
        // p = 2.0: Euclidean Distance
        else if (abs(p - 2.0f) < 1e-6f) {
            for (size_t i = 0; i < D; ++i) {
                float diff = static_cast<float>(x[i]) - static_cast<float>(y[i]);
                sum += diff * diff;
            }
            return sqrtf(fmaxf(0.0f, sum) + eps);
        }
        // p = inf: Chebyshev Distance
        else if (isinf(p)) {
            for (size_t i = 0; i < D; ++i) {
                float diff = fabsf(static_cast<float>(x[i]) - static_cast<float>(y[i]));
                if (diff > sum) sum = diff;
            }
            return sum; // Inf norm 不需要 eps 开方
        }
        // Generic p-norm
        else {
            for (size_t i = 0; i < D; ++i) {
                float diff = fabsf(static_cast<float>(x[i]) - static_cast<float>(y[i]));
                sum += powf(diff, p);
            }
            return powf(sum + eps, 1.0f / p);
        }
    }
};

// ==================================================================
// Kernel: Pointwise
// 每一个线程处理一个 Batch 样本 (One thread per sample N)
// 遍历维度 D 进行计算
// 输出 Tensor 形状 [N] 或 [N, 1] (内存布局一致)
// ==================================================================
template <typename T>
__global__ void pairwise_distance_kernel(
    T * __restrict__ output,        // [N]
    const T * __restrict__ x1,      // [N, D]
    const T * __restrict__ x2,      // [N, D]
    size_t N,
    size_t D,
    PairwiseDistanceFunctor functor) {

    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (n < N) {
        // 定位当前样本的起始位置
        const T* x1_ptr = x1 + n * D;
        const T* x2_ptr = x2 + n * D;

        // 计算距离
        float dist = functor.compute_dist(x1_ptr, x2_ptr, D);

        // 写入输出
        output[n] = static_cast<T>(dist);
    }
}

} // namespace op::pairwise_distance::cuda

#endif // __PAIRWISE_DISTANCE_CUDA_CUH__