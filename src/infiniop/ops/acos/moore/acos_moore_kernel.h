#ifndef __ACOS_MOORE_KERNEL_H__
#define __ACOS_MOORE_KERNEL_H__

/*
 * This file contains the Acos operation implementation for the MUSA backend.
 * Fixed: Reverted to standard ::acosf to avoid symbol resolution errors.
 */

namespace op::acos::moore {

typedef struct AcosOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &input) const {
        // -----------------------------------------------------------------
        // 1. Half2 (FP16x2)
        // -----------------------------------------------------------------
        if constexpr (std::is_same_v<T, half2>) {
            // MUSA 环境可能缺失 __h2acos，改用拆分转 float 处理
            float f1 = __low2float(input);
            float f2 = __high2float(input);

            // 使用标准库 ::acosf，编译器会处理 Device 端的映射
            return __floats2half2_rn(::acosf(f1), ::acosf(f2));
        } 
        // -----------------------------------------------------------------
        // 2. Half (FP16)
        // -----------------------------------------------------------------
        else if constexpr (std::is_same_v<T, half>) {
            // Half fallback to float
            float val_f = __half2float(input);
            return __float2half(::acosf(val_f));
        } 
        // -----------------------------------------------------------------
        // 3. Bfloat16
        // -----------------------------------------------------------------
        else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // BF16 fallback to float
            float val_f = __bfloat162float(input);
            return __float2bfloat16(::acosf(val_f));
        } 
        // -----------------------------------------------------------------
        // 4. Float32
        // -----------------------------------------------------------------
        else if constexpr (std::is_same_v<T, float>) {
            // 直接使用标准库 ::acosf
            return ::acosf(input);
        } 
        // -----------------------------------------------------------------
        // 5. Double / Other
        // -----------------------------------------------------------------
        else {
            return ::acos(input);
        }
    }
} AcosOp;
} // namespace op::acos::moore

#endif // __ACOS_MOORE_KERNEL_H__