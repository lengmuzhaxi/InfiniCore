#include "infinicore/ops/matrix_power.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

// 1. 初始化 Dispatcher
common::OpDispatcher<MatrixPower::schema> &MatrixPower::dispatcher() {
    static common::OpDispatcher<MatrixPower::schema> dispatcher_;
    return dispatcher_;
};

// 2. 类层面的 execute 方法
void MatrixPower::execute(Tensor output, Tensor input, int n) {
    // 切换上下文
    infinicore::context::setDevice(output->device());

    // 分发计算
    dispatcher().lookup(output->device().getType())(output, input, n);
}

// 3. Out-of-place 接口
Tensor matrix_power(Tensor input, int n) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    MatrixPower::execute(output, input, n);
    return output;
}

// 4. In-place 接口
void matrix_power_(Tensor output, Tensor input, int n) {
    MatrixPower::execute(output, input, n);
}

} // namespace infinicore::op