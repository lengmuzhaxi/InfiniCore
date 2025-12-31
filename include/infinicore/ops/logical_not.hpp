#pragma once
#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class LogicalNot {
public:
    using schema = void (*)(Tensor, Tensor);
    
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};
Tensor logical_not(Tensor input);
void logical_not_(Tensor output, Tensor input);

} // namespace infinicore::op