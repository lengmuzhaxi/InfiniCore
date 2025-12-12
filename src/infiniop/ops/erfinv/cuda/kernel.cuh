#ifndef __ERFINV_CUDA_H__
#define __ERFINV_CUDA_H__

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <math.h>
#include <type_traits>

namespace op::erfinv::cuda {

// ----------------------
// Fast erfinv wrapper
// ----------------------
__device__ __forceinline__ float fast_erfinvf(float x) {
    return erfinvf(x);
}

// ----------------------
// float kernel (F32)
// ----------------------
template<typename T>
__device__ __forceinline__ T erfinv_impl(T val);

template<>
__device__ __forceinline__ float erfinv_impl<float>(float val) {
    return fast_erfinvf(val);
}

// ----------------------
// half kernel (F16)
// ----------------------
template<>
__device__ __forceinline__ half erfinv_impl<half>(half val) {
#if (__CUDA_ARCH__ >= 530)
    float f = __half2float(val);
    return __float2half(fast_erfinvf(f));
#else
    float f = __half2float(val);
    return __float2half(fast_erfinvf(f));
#endif
}

// ----------------------
// half2 kernel (F16x2 vectorized)
// ----------------------
template<>
__device__ __forceinline__ half2 erfinv_impl<half2>(half2 val) {
#if (__CUDA_ARCH__ >= 530)
    float2 f = __half22float2(val);
    f.x = fast_erfinvf(f.x);
    f.y = fast_erfinvf(f.y);
    return __float22half2_rn(f);
#else
    float2 f = __half22float2(val);
    f.x = fast_erfinvf(f.x);
    f.y = fast_erfinvf(f.y);
    return __float22half2_rn(f);
#endif
}

// ----------------------
// bfloat16 kernel (BF16)
// ----------------------
template<>
__device__ __forceinline__ nv_bfloat16 erfinv_impl<nv_bfloat16>(nv_bfloat16 val) {
    float f = __bfloat162float(val);
    return __float2bfloat16(fast_erfinvf(f));
}

// ----------------------
// Fallback kernel
// ----------------------
template<typename T>
__device__ __forceinline__ T erfinv_impl(T val) {
    return static_cast<T>(fast_erfinvf(static_cast<float>(val)));
}

// ----------------------
// ErfinvOp struct
// ----------------------
struct ErfinvOp {
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &a) const {
        return erfinv_impl(a);
    }
};

} // namespace op::erfinv::cuda

#endif // __ERFINV_CUDA_H__