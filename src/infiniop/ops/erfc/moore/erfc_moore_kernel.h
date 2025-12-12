#ifndef __ERFC_MOORE_KERNEL_H__
#define __ERFC_MOORE_KERNEL_H__
#include <musa_fp16.h>
#include <musa_bf16.h>
#include <type_traits>
/*
 * This file contains the Erfc operation implementation for the MUSA backend.
 */

namespace op::erfc::moore {

typedef struct ErfcOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &input) const {
        // -----------------------------------------------------------------
        // 1. Half2 (FP16x2)
        // -----------------------------------------------------------------
        if constexpr (std::is_same_v<T, half2>) {
            
            float f1 = __low2float(input);
            float f2 = __high2float(input);
            // 使用 ::erfcf 计算 float 版本的 erfc
            return __floats2half2_rn(::erfcf(f1), ::erfcf(f2));
        } 
        // -----------------------------------------------------------------
        // 2. Half (FP16)
        // -----------------------------------------------------------------
        else if constexpr (std::is_same_v<T, half>) {
            // Half fallback to float
            float val_f = __half2float(input);
            return __float2half(::erfcf(val_f));
        } 
        // -----------------------------------------------------------------
        // 3. Bfloat16
        // -----------------------------------------------------------------
        else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
            // BF16 fallback to float
            float val_f = __bfloat162float(input);
            return __float2bfloat16(::erfcf(val_f));
        } 
        // -----------------------------------------------------------------
        // 4. Float32
        // -----------------------------------------------------------------
        else if constexpr (std::is_same_v<T, float>) {
            // 直接使用标准库 ::erfcf
            return ::erfcf(input);
        } 
        // -----------------------------------------------------------------
        // 5. Double / Other
        // -----------------------------------------------------------------
        else {
            return ::erfc(input);
        }
    }
} ErfcOp;
} // namespace op::erfc::moore

#endif // __ERFC_MOORE_KERNEL_H__