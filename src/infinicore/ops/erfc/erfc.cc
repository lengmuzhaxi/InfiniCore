#include "infinicore/ops/erfc.hpp"

namespace infinicore::op {

common::OpDispatcher<Erfc::schema> &Erfc::dispatcher() {
    static common::OpDispatcher<Erfc::schema> dispatcher_;
    return dispatcher_;
};

void Erfc::execute(Tensor output, Tensor input) {
    dispatcher().lookup(context::getDevice().getType())(output, input);
}

Tensor erfc(Tensor input) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    erfc_(output, input);
    return output;
}

void erfc_(Tensor output, Tensor input) {
    Erfc::execute(output, input);
}

} // namespace infinicore::op