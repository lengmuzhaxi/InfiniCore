#include "infinicore/ops/erfinv.hpp"

namespace infinicore::op {

common::OpDispatcher<Erfinv::schema> &Erfinv::dispatcher() {
    static common::OpDispatcher<Erfinv::schema> dispatcher_;
    return dispatcher_;
};

void Erfinv::execute(Tensor output, Tensor input) {
    dispatcher().lookup(context::getDevice().getType())(output, input);
}

Tensor erfinv(Tensor input) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    erfinv_(output, input);
    return output;
}

void erfinv_(Tensor output, Tensor input) {
    Erfinv::execute(output, input);
}

} // namespace infinicore::op