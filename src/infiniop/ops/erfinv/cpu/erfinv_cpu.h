#ifndef __ERFINV_CPU_H__
#define __ERFINV_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(erfinv, cpu)

#include <cmath>
#include <type_traits>
#include <limits>

namespace op::erfinv::cpu {

// --------------------------------------------------------------------------
// 手动实现 erfinv (Inverse Error Function)
// 算法来源: 基于 Winitzki 近似或类似的切比雪夫多项式逼近，
// 这里提供一个在深度学习框架中常用的高精度近似实现。
// --------------------------------------------------------------------------
template <typename T>
inline T my_erfinv(T x) {
    // 限制范围 (-1, 1)
    if (x <= -1.0 || x >= 1.0) {
        return std::numeric_limits<T>::infinity() * ((x > 0) ? 1 : -1);
    }

    // Winitzki 近似法 (相对误差 < 10^-4，对于 ML 足够，若需更高精度可使用更复杂的多项式)
    // 或者使用以下经典的近似公式（适用于 float/double）
    
    T tt = x * x;
    T lnx = -std::log(static_cast<T>(1.0) - tt);
    
    T a = static_cast<T>(0.147); 
    T pi = static_cast<T>(3.14159265358979323846);
    
    T term1 = static_cast<T>(2.0) / (pi * a) + lnx / static_cast<T>(2.0);
    T term2 = lnx / a;
    
    T result = std::sqrt(std::sqrt(term1 * term1 - term2) - term1);
    
    return (x >= 0) ? result : -result;
}

// --------------------------------------------------------------------------
// ErfinvOp 结构体
// --------------------------------------------------------------------------
typedef struct ErfinvOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        // 1. 整数类型：转 double 计算
        if constexpr (std::is_integral_v<T>) {
            return static_cast<T>(my_erfinv(static_cast<double>(x)));
        } 
        // 2. Float 类型
        else if constexpr (std::is_same_v<T, float>) {
            return my_erfinv<float>(x);
        }
        // 3. Double 类型
        else if constexpr (std::is_same_v<T, double>) {
            return my_erfinv<double>(x);
        } 
        // 4. 其他类型 (fp16, bf16) 转 float 计算
        else {
            return static_cast<T>(my_erfinv(static_cast<float>(x)));
        }
    }
} ErfinvOp;

} // namespace op::erfinv::cpu

#endif // __ERFINV_CPU_H__