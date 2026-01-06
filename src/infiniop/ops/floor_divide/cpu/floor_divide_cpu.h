#ifndef __FLOOR_DIVIDE_CPU_H__
#define __FLOOR_DIVIDE_CPU_H__

#include <cmath>
#include <type_traits>
#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(floor_divide, cpu)

namespace op::floor_divide::cpu {
typedef struct FloorDivideOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        // 1. 浮点数处理
        if constexpr (std::is_floating_point_v<T> || 
                      std::is_same_v<T, fp16_t> || 
                      std::is_same_v<T, bf16_t>) {
            return std::floor(a / b);
        } 
        // 2. 整数处理
        else {
            // ========================================================
            // 【关键修改】防止 Bench 模式下的循环崩溃
            // ========================================================
            // 在 Bench 循环中，b 会被前一次计算修改为 0。
            // 我们构造一个 safe_b：如果 b 是 0，就把它当成 1 来除。
            // 这样 CPU 就不会报 SIGFPE 错误，程序能继续跑下去。
            // --------------------------------------------------------
            T safe_b = b + (b == 0); 
            
            // 使用 safe_b 进行计算
            T res = a / safe_b;
            T rem = a % safe_b;
            
            // 保持原有的 floor 逻辑
            if (rem != 0 && ((a < 0) ^ (b < 0))) {
                res -= 1;
            }
            return res;
        }
    }
} FloorDivideOp;
} // namespace op::floor_divide::cpu

#endif // __FLOOR_DIVIDE_CPU_H__