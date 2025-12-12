#ifndef __ERF_CPU_H__
#define __ERF_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(erf, cpu)

#include <cmath>
#include <type_traits>

namespace op::erf::cpu {

typedef struct ErfOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        if constexpr (std::is_integral_v<T>) {
            return static_cast<T>(std::erf(static_cast<double>(x)));
        } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            return std::erf(x);
        } else {
            return static_cast<T>(std::erf(static_cast<float>(x)));
        }
    }
} ErfOp;

} // namespace op::erf::cpu

#endif // __ERF_CPU_H__