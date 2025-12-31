#include "common/op.hpp"

namespace infinicore::op {

class PairwiseDistance {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, float, float, bool);
    
    // 修改: execute 函数签名增加 keepdim
    static void execute(Tensor output, Tensor x1, Tensor x2, float p, float eps, bool keepdim);
    
    static common::OpDispatcher<schema> &dispatcher();
};
Tensor pairwise_distance(Tensor x1, Tensor x2, float p = 2.0f, float eps = 1e-6f, bool keepdim = false);

// In-place / Explicit Output API
// 修改: 增加 keepdim
void pairwise_distance_(Tensor output, Tensor x1, Tensor x2, float p, float eps, bool keepdim);

} // namespace infinicore::op