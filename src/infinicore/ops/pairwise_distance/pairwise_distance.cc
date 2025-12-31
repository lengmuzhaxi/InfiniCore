#include "infinicore/ops/pairwise_distance.hpp"

namespace infinicore::op {

// 1. 定义 Dispatcher 单例
common::OpDispatcher<PairwiseDistance::schema> &PairwiseDistance::dispatcher() {
    static common::OpDispatcher<PairwiseDistance::schema> dispatcher_;
    return dispatcher_;
};

// 2. Execute 静态方法
void PairwiseDistance::execute(Tensor output, Tensor x1, Tensor x2, float p, float eps, bool keepdim) {
    Tensor x1_c = x1->is_contiguous() ? x1 : x1->contiguous();
    Tensor x2_c = x2->is_contiguous() ? x2 : x2->contiguous();

    // 将处理后的连续 Tensor 传给后端
    dispatcher().lookup(context::getDevice().getType())(output, x1_c, x2_c, p, eps, keepdim);
}

// 3. 函数式接口 (Functional API)
Tensor pairwise_distance(Tensor x1, Tensor x2, float p, float eps, bool keepdim) {
    // 推导输出形状
    Shape output_shape = x1->shape();

    if (!output_shape.empty()) {
        if (keepdim) {
            // 保留维度: (..., D) -> (..., 1)
            output_shape.back() = 1;
        } else {
            // 不保留维度: (..., D) -> (...)
            output_shape.pop_back();
        }
    }

    // 创建输出 Tensor
    auto output = Tensor::empty(output_shape, x1->dtype(), x1->device());
    
    // 调用 In-place 接口执行
    pairwise_distance_(output, x1, x2, p, eps, keepdim);
    return output;
}

// 4. In-place / Explicit Output API
void pairwise_distance_(Tensor output, Tensor x1, Tensor x2, float p, float eps, bool keepdim) {
    PairwiseDistance::execute(output, x1, x2, p, eps, keepdim);
}

} // namespace infinicore::op