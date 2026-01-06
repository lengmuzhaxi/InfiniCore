#include "infinicore/ops/triplet_margin_loss.hpp"

namespace infinicore::op {

// 1. 定义 Dispatcher 单例 (保持不变)
common::OpDispatcher<TripletMarginLoss::schema> &TripletMarginLoss::dispatcher() {
    static common::OpDispatcher<TripletMarginLoss::schema> dispatcher_;
    return dispatcher_;
};

void TripletMarginLoss::execute(Tensor output, Tensor anchor, Tensor positive, Tensor negative, float margin, int64_t p, float eps, bool swap, int64_t reduction) {
    dispatcher().lookup(context::getDevice().getType())(output, anchor, positive, negative, margin, p, eps, swap, reduction);
}

// 3. 函数式接口 (修改这里！)
Tensor triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, float margin, int64_t p, float eps, bool swap, int64_t reduction) {
    Shape output_shape;
    if (reduction == 0) { // None
        output_shape = {anchor->shape()[0]};
    } else { 
        output_shape = {}; // Scalar
    }

    // 【新增关键步骤】强制输入连续化
    // 这一步至关重要！它确保 anchor/pos/neg 在内存中是紧凑排列的 (N, D)。
    // 否则 CPU Kernel 按 ptr + n*D 访问会读到错误数据。
    auto anchor_contig = anchor->contiguous();
    auto positive_contig = positive->contiguous();
    auto negative_contig = negative->contiguous();

    // 使用 anchor 的属性创建输出 Tensor
    auto output = Tensor::empty(output_shape, anchor->dtype(), anchor->device());
    
    // 调用 execute 时传入连续化后的 Tensor
    triplet_margin_loss_(output, anchor_contig, positive_contig, negative_contig, margin, p, eps, swap, reduction);
    return output;
}

void triplet_margin_loss_(Tensor output, Tensor anchor, Tensor positive, Tensor negative, float margin, int64_t p, float eps, bool swap, int64_t reduction) {
    TripletMarginLoss::execute(output, anchor, positive, negative, margin, p, eps, swap, reduction);
}

} // namespace infinicore::op