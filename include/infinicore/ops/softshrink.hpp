#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Softshrink {
public:
    using schema = void (*)(Tensor, Tensor, float);
    static void execute(Tensor output, Tensor input, float lambda);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor softshrink(Tensor input, float lambda = 0.5f);
void softshrink_(Tensor output, Tensor input, float lambda = 0.5f);

} // namespace infinicore::op
