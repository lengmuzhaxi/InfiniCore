#ifndef __ERF_CUDA_H__
#define __ERF_CUDA_H__

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <math.h>
#include <type_traits>

namespace op::erf::cuda {

// ----------------------
// Fast erf wrapper
// ----------------------
__device__ __forceinline__ float fast_erff(float x) {
    return erff(x);
}

// ----------------------
// float kernel (F32)
// ----------------------
template<typename T>
__device__ __forceinline__ T erf_impl(T val);

template<>
__device__ __forceinline__ float erf_impl<float>(float val) {
    return fast_erff(val);
}

// ----------------------
// half kernel (F16)
// ----------------------
template<>
__device__ __forceinline__ half erf_impl<half>(half val) {
#if (__CUDA_ARCH__ >= 530)
    float f = __half2float(val);
    return __float2half(fast_erff(f));
#else
    float f = __half2float(val);
    return __float2half(fast_erff(f));
#endif
}

// ----------------------
// half2 kernel (F16x2 vectorized)
// ----------------------
template<>
__device__ __forceinline__ half2 erf_impl<half2>(half2 val) {
#if (__CUDA_ARCH__ >= 530)
    float2 f = __half22float2(val);
    f.x = fast_erff(f.x);
    f.y = fast_erff(f.y);
    return __float22half2_rn(f);
#else
    float2 f = __half22float2(val);
    f.x = fast_erff(f.x);
    f.y = fast_erff(f.y);
    return __float22half2_rn(f);
#endif
}

// ----------------------
// bfloat16 kernel (BF16)
// ----------------------
template<>
__device__ __forceinline__ nv_bfloat16 erf_impl<nv_bfloat16>(nv_bfloat16 val) {
    float f = __bfloat162float(val);
    return __float2bfloat16(fast_erff(f));
}

// ----------------------
// Fallback kernel
// ----------------------
template<typename T>
__device__ __forceinline__ T erf_impl(T val) {
    return static_cast<T>(fast_erff(static_cast<float>(val)));
}

// ----------------------
// ErfOp struct
// ----------------------
struct ErfOp {
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &a) const {
        return erf_impl(a);
    }
};

} // namespace op::erf::cuda

#endif // __ERF_CUDA_H__