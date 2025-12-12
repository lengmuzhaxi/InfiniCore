#ifndef __ERFC_CUDA_H__
#define __ERFC_CUDA_H__

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <math.h>
#include <type_traits>

namespace op::erfc::cuda {

// ----------------------
// Fast erfc wrapper
// ----------------------
__device__ __forceinline__ float fast_erfcf(float x) {
    return erfcf(x);
}

// ----------------------
// float kernel (F32)
// ----------------------
template<typename T>
__device__ __forceinline__ T erfc_impl(T val);

template<>
__device__ __forceinline__ float erfc_impl<float>(float val) {
    return fast_erfcf(val);
}

// ----------------------
// half kernel (F16)
// ----------------------
template<>
__device__ __forceinline__ half erfc_impl<half>(half val) {
#if (__CUDA_ARCH__ >= 530)
    float f = __half2float(val);
    return __float2half(fast_erfcf(f));
#else
    float f = __half2float(val);
    return __float2half(fast_erfcf(f));
#endif
}

// ----------------------
// half2 kernel (F16x2 vectorized)
// ----------------------
template<>
__device__ __forceinline__ half2 erfc_impl<half2>(half2 val) {
#if (__CUDA_ARCH__ >= 530)
    float2 f = __half22float2(val);
    f.x = fast_erfcf(f.x);
    f.y = fast_erfcf(f.y);
    return __float22half2_rn(f);
#else
    float2 f = __half22float2(val);
    f.x = fast_erfcf(f.x);
    f.y = fast_erfcf(f.y);
    return __float22half2_rn(f);
#endif
}

// ----------------------
// bfloat16 kernel (BF16)
// ----------------------
template<>
__device__ __forceinline__ nv_bfloat16 erfc_impl<nv_bfloat16>(nv_bfloat16 val) {
    float f = __bfloat162float(val);
    return __float2bfloat16(fast_erfcf(f));
}

// ----------------------
// Fallback kernel
// ----------------------
template<typename T>
__device__ __forceinline__ T erfc_impl(T val) {
    return static_cast<T>(fast_erfcf(static_cast<float>(val)));
}

// ----------------------
// ErfcOp struct
// ----------------------
struct ErfcOp {
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &a) const {
        return erfc_impl(a);
    }
};

} // namespace op::erfc::cuda

#endif // __ERFC_CUDA_H__