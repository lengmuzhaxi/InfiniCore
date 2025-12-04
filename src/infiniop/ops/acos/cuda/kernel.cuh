#ifndef __ACOS_CUDA_H__
#define __ACOS_CUDA_H__

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath> // 必须包含 cmath 以使用 acosf

namespace op::acos::cuda {

typedef struct AcosOp {
public:
    // Acos 是一元算子
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &a) const {
        // ==========================================
        // 1. Half2 (FP16x2)
        // ==========================================
        if constexpr (std::is_same_v<T, half2>) {
            // 兼容写法：解包成 float2 -> 分别计算 -> 打包回 half2
            // 解决旧版 CUDA 没有 h2acos 的问题
            float2 f2 = __half22float2(a);
            f2.x = acosf(f2.x);
            f2.y = acosf(f2.y);
            return __float22half2_rn(f2);
        } 
        // ==========================================
        // 2. Half (FP16)
        // ==========================================
        else if constexpr (std::is_same_v<T, half>) {
            // 兼容写法：转 float -> 计算 -> 转回 half
            // 解决旧版 CUDA 没有 hacos 的问题
            return __float2half(acosf(__half2float(a)));
        } 
        // ==========================================
        // 3. BFloat16
        // ==========================================
        else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // BF16 必须转 float 计算
            return __float2bfloat16(acosf(__bfloat162float(a)));
        } 
        // ==========================================
        // 4. Float (FP32)
        // ==========================================
        else if constexpr (std::is_same_v<T, float>) {
            return acosf(a);
        } 
        // ==========================================
        // 5. Double / Standard
        // ==========================================
        else {
            return ::acos(a);
        }
    }
} AcosOp;

} // namespace op::acos::cuda

#endif // __ACOS_CUDA_H__