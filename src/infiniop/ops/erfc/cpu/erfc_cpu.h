#ifndef __ERFC_CPU_H__
#define __ERFC_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(erfc, cpu)

#include <cmath>
#include <type_traits>

namespace op::erfc::cpu {

typedef struct ErfcOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        if constexpr (std::is_integral_v<T>) {
            return static_cast<T>(std::erfc(static_cast<double>(x)));
        } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            return std::erfc(x);
        } else {
            return static_cast<T>(std::erfc(static_cast<float>(x)));
        }
    }
} ErfcOp;

} // namespace op::erfc::cpu

#endif // __ERFC_CPU_H__