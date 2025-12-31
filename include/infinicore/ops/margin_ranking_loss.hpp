#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class MarginRankingLoss {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, float, int64_t, int64_t);
    
    static void execute(Tensor output, Tensor input1, Tensor input2, Tensor target, float margin, int64_t p, int64_t reduction);
    
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, float margin = 0.0f, int64_t p = 1, int64_t reduction = 1);

void margin_ranking_loss_(Tensor output, Tensor input1, Tensor input2, Tensor target, float margin, int64_t p, int64_t reduction);

} // namespace infinicore::op