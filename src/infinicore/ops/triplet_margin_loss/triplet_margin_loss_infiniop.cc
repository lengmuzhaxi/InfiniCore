#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/triplet_margin_loss.hpp"
#include <infiniop.h>

namespace infinicore::op::triplet_margin_loss_impl::infiniop {

// 定义描述符缓存
thread_local common::OpCache<size_t, infiniopTripletMarginLossDescriptor_t> caches(
    100, // capacity
    [](infiniopTripletMarginLossDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyTripletMarginLossDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor anchor, Tensor positive, Tensor negative, float margin, int64_t p, float eps, bool swap, int64_t reduction) {
    
    // 【修改点 1】强制 Tensor 连续化
    // 这解决了因 Stride (如 strides=(40, 5)) 导致的指针访问越界或数据偏移错误
    auto anchor_c = anchor->contiguous();
    auto positive_c = positive->contiguous();
    auto negative_c = negative->contiguous();

    // 1. 计算 Hash Seed 作为 Cache Key
    // 确保包含所有标量参数，防止不同 eps 的用例误命中同一个 Descriptor
    size_t seed = hash_combine(output, anchor_c, positive_c, negative_c, margin, p, eps, swap, reduction);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopTripletMarginLossDescriptor_t desc = nullptr;

    if (!desc_opt) {
        // 2. 创建描述符
        // 使用连续化后的描述符创建
        INFINICORE_CHECK_ERROR(infiniopCreateTripletMarginLossDescriptor(
            context::getInfiniopHandle(output->device()), 
            &desc,
            output->desc(), 
            anchor_c->desc(), 
            positive_c->desc(), 
            negative_c->desc(), 
            margin,
            static_cast<int>(p),
            eps,
            static_cast<int>(swap), 
            static_cast<int>(reduction)
        ));
        
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // 3. 获取 Workspace 并执行
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetTripletMarginLossWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    // 【修改点 2】执行时传入连续化后的数据指针 (anchor_c->data() 等)
    INFINICORE_CHECK_ERROR(infiniopTripletMarginLoss(
        desc, 
        workspace->data(), 
        workspace_size,
        output->data(), 
        anchor_c->data(), 
        positive_c->data(), 
        negative_c->data(),
        context::getStream()
    ));
}

static bool registered = []() {
    TripletMarginLoss::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::triplet_margin_loss_impl::infiniop