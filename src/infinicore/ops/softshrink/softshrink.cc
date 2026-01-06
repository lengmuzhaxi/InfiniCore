#include "infinicore/ops/softshrink.hpp"

namespace infinicore::op {

common::OpDispatcher<Softshrink::schema> &Softshrink::dispatcher() {
    static common::OpDispatcher<Softshrink::schema> dispatcher_;
    return dispatcher_;
}

void Softshrink::execute(Tensor output, Tensor input, float lambda) {
    dispatcher().lookup(context::getDevice().getType())(output, input, lambda);
}

Tensor softshrink(Tensor input, float lambda) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    softshrink_(output, input, lambda);
    return output;
}

void softshrink_(Tensor output, Tensor input, float lambda) {
    Softshrink::execute(output, input, lambda);
}

} // namespace infinicore::op
