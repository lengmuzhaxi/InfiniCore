#ifndef __ADAPTIVE_AVG_POOL1D_MOORE_KERNEL_H__
#define __ADAPTIVE_AVG_POOL1D_MOORE_KERNEL_H__

#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h> // 必须包含，用于 __mt_bfloat16 定义和 intrinsics

#include <type_traits> // 用于 std::is_same_v

namespace op::adaptive_avg_pool1d::moore {

typedef struct AdaptiveAvgPool1dOp {
public:
    template <typename T>
    __device__ __forceinline__ void operator()(
        const int w_out,       // 当前输出在 L 维度的索引
        const int input_size,  // L_in
        const int output_size, // L_out
        const T* input_base,   // 当前通道输入的起始地址 (ptr + c * L_in)
        T* output_ptr          // 当前输出元素的写入地址
    ) const {
        
        // 1. 计算池化窗口的起始和结束索引 (Start & End)
        // 映射公式: 
        // start = floor(w_out * L_in / L_out)
        // end   = ceil((w_out + 1) * L_in / L_out)
        
        // 使用整数运算实现 ceil: ceil(a/b) = (a + b - 1) / b
        int start = (w_out * input_size) / output_size;
        int end = ((w_out + 1) * input_size + output_size - 1) / output_size;

        // 边界保护 (Defensive check)
        start = (start < 0) ? 0 : start;
        end = (end > input_size) ? input_size : end;

        int kernel_size = end - start;
        // 避免除以 0 (虽然逻辑上 valid input 不会触发，但在 GPU 上需谨慎)
        kernel_size = (kernel_size < 1) ? 1 : kernel_size;

        // 2. 累加逻辑 (Accumulation)
        // 强制使用 float 进行累加，防止 FP16/BF16 精度溢出
        float sum = 0.0f;

        for (int i = start; i < end; ++i) {
            T val = input_base[i];

            if constexpr (std::is_same_v<T, half>) {
                // FP16 -> Float
                sum += __half2float(val);
            } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
                // BF16 -> Float (使用 MUSA 专用 intrinsic)
                sum += __bfloat162float(val);
            } else {
                // Float/Double
                sum += static_cast<float>(val);
            }
        }

        // 3. 平均值计算与写回 (Average & Write Back)
        float avg = sum / static_cast<float>(kernel_size);

        if constexpr (std::is_same_v<T, half>) {
            // Float -> FP16
            *output_ptr = __float2half(avg);
        } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
            // Float -> BF16 (使用 MUSA 专用 intrinsic)
            *output_ptr = __float2bfloat16(avg);
        } else {
            // Float/Double
            *output_ptr = static_cast<T>(avg);
        }
    }

} AdaptiveAvgPool1dOp;

} // namespace op::adaptive_avg_pool1d::moore

#endif // __ADAPTIVE_AVG_POOL1D_MOORE_KERNEL_H__