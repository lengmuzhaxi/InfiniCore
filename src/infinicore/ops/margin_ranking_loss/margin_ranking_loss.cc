#include "infinicore/ops/margin_ranking_loss.hpp"
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <string>

namespace infinicore::op {

// ========================================================================
// 0. 内部辅助函数：手动实现形状广播推导 (与 Lerp 保持一致)
// ========================================================================
namespace {

Shape compute_broadcast_shape(const std::vector<Shape>& shapes) {
    if (shapes.empty()) return {};
    
    size_t max_ndim = 0;
    for (const auto& shape : shapes) {
        max_ndim = std::max(max_ndim, shape.size());
    }

    Shape out_shape(max_ndim);

    for (size_t i = 0; i < max_ndim; ++i) {
        size_t current_dim_val = 1;
        bool set = false;

        for (const auto& shape : shapes) {
            if (i < shape.size()) {
                size_t dim = shape[shape.size() - 1 - i];
                if (dim == 1) continue; 

                if (!set) {
                    current_dim_val = dim;
                    set = true;
                } else if (current_dim_val != dim) {
                    throw std::runtime_error(
                        "MarginRankingLoss: Shapes are not broadcastable. Mismatch at dimension offset " + 
                        std::to_string(i));
                }
            }
        }
        out_shape[max_ndim - 1 - i] = current_dim_val;
    }
    return out_shape;
}

} // namespace anonymous

// ========================================================================
// 1. 定义 Dispatcher 单例
// ========================================================================
common::OpDispatcher<MarginRankingLoss::schema> &MarginRankingLoss::dispatcher() {
    static common::OpDispatcher<MarginRankingLoss::schema> dispatcher_;
    return dispatcher_;
};

// ========================================================================
// 2. Execute 静态方法
// ========================================================================
// 修改：增加了 int64_t p
void MarginRankingLoss::execute(Tensor output, Tensor input1, Tensor input2, Tensor target, float margin, int64_t p, int64_t reduction) {
    dispatcher().lookup(context::getDevice().getType())(output, input1, input2, target, margin, p, reduction);
}

// ========================================================================
// 3. 函数式接口
// ========================================================================
// 修改：增加了 int64_t p
Tensor margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, float margin, int64_t p, int64_t reduction) {
    Shape output_shape;
    
    if (reduction == 0) { // None
        // 修改：使用广播推导计算输出形状，而不是简单假设等于 input1
        output_shape = compute_broadcast_shape({
            input1->shape(),
            input2->shape(),
            target->shape()
        });
    } else { 
        output_shape = {}; // Scalar (Mean/Sum)
    }

    auto output = Tensor::empty(output_shape, input1->dtype(), input1->device());
    
    margin_ranking_loss_(output, input1, input2, target, margin, p, reduction);
    return output;
}

// ========================================================================
// 4. In-place / Explicit Output API
// ========================================================================
// 修改：增加了 int64_t p
void margin_ranking_loss_(Tensor output, Tensor input1, Tensor input2, Tensor target, float margin, int64_t p, int64_t reduction) {
    MarginRankingLoss::execute(output, input1, input2, target, margin, p, reduction);
}

} // namespace infinicore::op