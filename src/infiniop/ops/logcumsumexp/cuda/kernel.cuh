#ifndef __LOGCUMSUMEXP_CUDA_CUH__
#define __LOGCUMSUMEXP_CUDA_CUH__

#include <cuda_runtime.h>
#if defined ENABLE_METAX_API
    #include <maca_fp16.h>
    #include <maca_bfloat16.h>
#else
    #include <cuda_fp16.h>
    #include <cuda_bf16.h>
#endif

#include <cmath>

namespace op::logcumsumexp::cuda {

// ============================================================
// 数值稳定 log-sum-exp prefix state
// ============================================================

struct LSEState {
    float m;
    float s;

    __device__ __forceinline__ static LSEState identity() {
        return {-1e38f, 0.0f};
    }

    __device__ __forceinline__ void update(float v) {
        float nm = fmaxf(m, v);
        s = s * expf(m - nm) + expf(v - nm);
        m = nm;
    }

    __device__ __forceinline__ float value() const {
        return (s <= 0.f) ? -1e38f : m + logf(s);
    }
};

// ============================================================
// kernel：一个 thread = 一个 (outer, inner) 向量
// ============================================================

template <typename T>
__global__ void logcumsumexp_kernel(
    T* __restrict__ y,
    const T* __restrict__ x,
    size_t outer_size,
    size_t axis_size,
    size_t inner_size,

    size_t x_axis_stride,
    size_t x_inner_stride,
    size_t x_outer_stride,

    size_t y_axis_stride,
    size_t y_inner_stride,
    size_t y_outer_stride,

    bool exclusive,
    bool reverse) {

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_vec = outer_size * inner_size;
    if (tid >= num_vec) return;

    size_t o = tid / inner_size;
    size_t i = tid % inner_size;

    // ✅ 正确的 base offset（不再猜）
    size_t x_base = o * x_outer_stride + i * x_inner_stride;
    size_t y_base = o * y_outer_stride + i * y_inner_stride;

    LSEState state = LSEState::identity();

    for (size_t k = 0; k < axis_size; ++k) {
        size_t kk = reverse ? (axis_size - 1 - k) : k;

        size_t x_off = x_base + kk * x_axis_stride;
        size_t y_off = y_base + kk * y_axis_stride;

        float v = static_cast<float>(x[x_off]);

        if (exclusive) {
            y[y_off] = static_cast<T>(state.value());
            state.update(v);
        } else {
            state.update(v);
            y[y_off] = static_cast<T>(state.value());
        }
    }
}

} // namespace op::logcumsumexp::cuda

#endif
