#ifndef __SOFTSHRINK_CUDA_KERNEL_CUH__
#define __SOFTSHRINK_CUDA_KERNEL_CUH__

#include <cuda_runtime.h>
#include <cmath>
#include <type_traits>

#if defined ENABLE_METAX_API
    #include <maca_fp16.h>
    #include <maca_bfloat16.h>
    using nv_bfloat162 = __maca_bfloat162;
    using nv_bfloat16  = __maca_bfloat16;
#else
    #include <cuda_fp16.h>
    #include <cuda_bf16.h>
#endif

namespace op::softshrink::cuda {

struct SoftshrinkOp {
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x, const float lambda) const {
        if constexpr (std::is_same_v<T, half2>) {
            float2 vf = __half22float2(x);
            float2 vr;
            vr.x = (vf.x > lambda) ? (vf.x - lambda) : ((vf.x < -lambda) ? (vf.x + lambda) : 0.0f);
            vr.y = (vf.y > lambda) ? (vf.y - lambda) : ((vf.y < -lambda) ? (vf.y + lambda) : 0.0f);
            return __float22half2_rn(vr);
        } else if constexpr (std::is_same_v<T, nv_bfloat162>) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
                float f0 = __bfloat162float(__low2bfloat16(x));
                float f1 = __bfloat162float(__high2bfloat16(x));
                float r0 = (f0 > lambda) ? (f0 - lambda) : ((f0 < -lambda) ? (f0 + lambda) : 0.0f);
                float r1 = (f1 > lambda) ? (f1 - lambda) : ((f1 < -lambda) ? (f1 + lambda) : 0.0f);
                return __floats2bfloat162_rn(r0, r1);
            #else
                return x;
            #endif
        } else if constexpr (std::is_same_v<T, nv_bfloat16>) { // 这里使用了 nv_bfloat16
            float f = __bfloat162float(x);
            float r = (f > lambda) ? (f - lambda) : ((f < -lambda) ? (f + lambda) : 0.0f);
            return __float2bfloat16(r);
        } else if constexpr (std::is_same_v<T, half>) {
            float f = __half2float(x);
            float r = (f > lambda) ? (f - lambda) : ((f < -lambda) ? (f + lambda) : 0.0f);
            return __float2half(r);
        } else if constexpr (std::is_same_v<T, float>) {
            return (x > lambda) ? (x - lambda) : ((x < -lambda) ? (x + lambda) : 0.0f);
        } else if constexpr (std::is_same_v<T, double>) {
            return (x > (double)lambda) ? (x - (double)lambda) : ((x < -(double)lambda) ? (x + (double)lambda) : 0.0);
        } else if constexpr (std::is_integral_v<T>) {
            return (x > (T)lambda) ? (x - (T)lambda) : ((x < (T)-(lambda)) ? (x + (T)lambda) : (T)0);
        } else {
            auto fx = (double)x;
            double r = (fx > (double)lambda) ? (fx - (double)lambda) : ((fx < -(double)lambda) ? (fx + (double)lambda) : 0.0);
            return (T)r;
        }
    }
};

}

#endif