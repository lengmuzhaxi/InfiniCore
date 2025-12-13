#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class MatrixPower {
public:
    // 参数: output, input, n
    using schema = void (*)(Tensor, Tensor, int);
    static void execute(Tensor output, Tensor input, int n);
    
    static common::OpDispatcher<schema> &dispatcher();
};

// Out-of-place: matrix_power(input, n) -> Tensor
Tensor matrix_power(Tensor input, int n);

// In-place-like: matrix_power_(output, input, n)
void matrix_power_(Tensor output, Tensor input, int n);

} // namespace infinicore::op