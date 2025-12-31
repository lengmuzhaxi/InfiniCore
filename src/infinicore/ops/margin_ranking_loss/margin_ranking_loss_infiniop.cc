#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/margin_ranking_loss.hpp"
#include <infiniop.h>

namespace infinicore::op::margin_ranking_loss_impl::infiniop {

// 定义描述符缓存
thread_local common::OpCache<size_t, infiniopMarginRankingLossDescriptor_t> caches(
    100, // capacity
    [](infiniopMarginRankingLossDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyMarginRankingLossDescriptor(desc));
            desc = nullptr;
        }
    });

// 修改：增加了 int64_t p 参数
void calculate(Tensor output, Tensor input1, Tensor input2, Tensor target, float margin, int64_t p, int64_t reduction) {
    // 1. 计算 Hash Seed
    // 修改：将 p 加入哈希计算
    size_t seed = hash_combine(output, input1, input2, target, margin, p, reduction);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopMarginRankingLossDescriptor_t desc = nullptr;

    if (!desc_opt) {
        // 2. 创建描述符
        INFINICORE_CHECK_ERROR(infiniopCreateMarginRankingLossDescriptor(
            context::getInfiniopHandle(output->device()), 
            &desc,
            output->desc(), 
            input1->desc(), 
            input2->desc(), 
            target->desc(), 
            margin,
            static_cast<int>(p), // 修改：传递 p 参数
            static_cast<int>(reduction)
        ));
        
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // 3. 获取 Workspace 并执行
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetMarginRankingLossWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopMarginRankingLoss(
        desc, 
        workspace->data(), 
        workspace_size,
        output->data(), 
        input1->data(), 
        input2->data(), 
        target->data(), 
        context::getStream()
    ));
}

static bool registered = []() {
    MarginRankingLoss::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::margin_ranking_loss_impl::infiniop