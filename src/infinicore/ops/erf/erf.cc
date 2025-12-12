#include "infinicore/ops/erf.hpp"

namespace infinicore::op {

common::OpDispatcher<Erf::schema> &Erf::dispatcher() {
    static common::OpDispatcher<Erf::schema> dispatcher_;
    return dispatcher_;
};

void Erf::execute(Tensor output, Tensor input) {
    dispatcher().lookup(context::getDevice().getType())(output, input);
}

Tensor erf(Tensor input) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    erf_(output, input);
    return output;
}

void erf_(Tensor output, Tensor input) {
    Erf::execute(output, input);
}

} // namespace infinicore::op