#ifndef __INDEX_ADD_MOORE_KERNEL_H__
#define __INDEX_ADD_MOORE_KERNEL_H__

#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>
#include <type_traits>

namespace op::index_add::moore {

// ==================================================================
// 1. 原子加辅助函数 (Atomic Add Helper)
//    MUSA 可能未在所有架构上提供 half/bf16 的原生 atomicAdd，
//    因此使用 atomicCAS 实现通用的软原子操作。
// ==================================================================

// 基础模板：对于 float/double/int32，MUSA 通常支持原生 atomicAdd
template <typename T>
__device__ __forceinline__ void atomic_add_func(T* address, T val) {
    atomicAdd(address, val);
}

// ------------------------------------------------------------------
// 特化：Half (FP16)
// 使用 unsigned short (16bit) 进行 CAS 循环
// ------------------------------------------------------------------
template <>
__device__ __forceinline__ void atomic_add_func<half>(half* address, half val) {
    // 将地址重解释为 unsigned short* 以便进行位操作
    unsigned short* address_as_us = reinterpret_cast<unsigned short*>(address);
    unsigned short old = *address_as_us;
    unsigned short assumed;

    do {
        assumed = old;
        // 1. 将 old (16bit) 转为 float
        // 2. 将 val (16bit) 转为 float
        // 3. float 相加
        // 4. 结果转回 half (16bit)
        // 5. 重新打包为 unsigned short 用于 CAS
        half sum = __float2half(__half2float(*reinterpret_cast<const half*>(&assumed)) + 
                                __half2float(val));
        
        unsigned short sum_as_us = *reinterpret_cast<unsigned short*>(&sum);

        // CAS: 如果 *address == assumed，则写入 sum_as_us，否则返回 *address 的当前值
        old = atomicCAS(address_as_us, assumed, sum_as_us);

    } while (assumed != old); // 如果 old != assumed，说明由于竞争写入失败，重试
}

// ------------------------------------------------------------------
// 特化：BFloat16 (__mt_bfloat16)
// 同样使用 unsigned short (16bit) 进行 CAS 循环
// ------------------------------------------------------------------
template <>
__device__ __forceinline__ void atomic_add_func<__mt_bfloat16>(__mt_bfloat16* address, __mt_bfloat16 val) {
    unsigned short* address_as_us = reinterpret_cast<unsigned short*>(address);
    unsigned short old = *address_as_us;
    unsigned short assumed;

    do {
        assumed = old;
        // BF16 -> Float -> Add -> BF16
        float sum_f = __bfloat162float(*reinterpret_cast<const __mt_bfloat16*>(&assumed)) + 
                      __bfloat162float(val);
        
        __mt_bfloat16 sum_bf = __float2bfloat16(sum_f);
        unsigned short sum_as_us = *reinterpret_cast<unsigned short*>(&sum_bf);

        old = atomicCAS(address_as_us, assumed, sum_as_us);

    } while (assumed != old);
}

// ==================================================================
// 2. Kernel Functor
// ==================================================================

typedef struct IndexAddOp {
public:
    template <typename T, typename TIdx>
    __device__ __forceinline__ void operator()(
        const size_t curr_idx,      // Flattened index for Source
        const size_t index_len,     // Length of Index tensor
        const size_t inner_size,    // Stride of inner dims
        const size_t dim_size,      // Size of target dim in Output
        const float alpha,          // Scale factor
        const T* source,            // Source Tensor
        const TIdx* indices,        // Index Tensor
        T* output                   // Output Tensor
    ) const {
        
        // --- 1. 坐标映射 (Flat Index -> Multi-dim Index) ---
        // source_idx -> (outer, idx_in_index, inner)
        
        size_t inner_idx = curr_idx % inner_size;
        size_t tmp = curr_idx / inner_size;
        size_t idx_in_indices = tmp % index_len; // 当前处理的是 Index 张量中的第几个索引
        size_t outer_idx = tmp / index_len;

        // --- 2. 读取 Source 并应用 Alpha ---
        T src_val = source[curr_idx];
        float val_f;

        // 统一转 float 计算乘法
        if constexpr (std::is_same_v<T, half>) {
            val_f = __half2float(src_val);
        } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
            val_f = __bfloat162float(src_val);
        } else {
            val_f = static_cast<float>(src_val);
        }

        val_f *= alpha;

        // 转回 T
        T add_val;
        if constexpr (std::is_same_v<T, half>) {
            add_val = __float2half(val_f);
        } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
            add_val = __float2bfloat16(val_f);
        } else {
            add_val = static_cast<T>(val_f);
        }

        // --- 3. 读取 Index 并计算 Output 偏移 ---
        TIdx target_dim_idx = indices[idx_in_indices];

        // 处理 Python 风格负索引
        if (target_dim_idx < 0) {
            target_dim_idx += static_cast<TIdx>(dim_size);
        }

        // --- 4. 边界检查 & 原子累加 ---
        if (target_dim_idx >= 0 && target_dim_idx < static_cast<TIdx>(dim_size)) {
            // output_offset = outer * (dim_size * inner) + target_idx * inner + inner
            size_t out_offset = outer_idx * (dim_size * inner_size) + 
                                static_cast<size_t>(target_dim_idx) * inner_size + 
                                inner_idx;
            
            // 调用我们手写的 CAS 原子加
            atomic_add_func(output + out_offset, add_val);
        }
    }

} IndexAddOp;

} // namespace op::index_add::moore

#endif // __INDEX_ADD_MOORE_KERNEL_H__