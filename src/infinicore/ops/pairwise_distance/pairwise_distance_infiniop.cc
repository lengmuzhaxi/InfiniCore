#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/pairwise_distance.hpp"
#include <infiniop.h>

namespace infinicore::op::pairwise_distance_impl::infiniop {

// 定义描述符缓存
// 缓存 Key 为 size_t (Hash值)，Value 为 infiniopPairwiseDistanceDescriptor_t
thread_local common::OpCache<size_t, infiniopPairwiseDistanceDescriptor_t> caches(
    100, // capacity
    [](infiniopPairwiseDistanceDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyPairwiseDistanceDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor x1, Tensor x2, float p, float eps, bool keepdim) {
    // 1. 计算 Hash Seed 作为 Cache Key
    // 将 keepdim 加入哈希计算，因为不同的 keepdim 对应不同的 descriptor 属性
    size_t seed = hash_combine(output, x1, x2, p, eps, keepdim);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopPairwiseDistanceDescriptor_t desc = nullptr;

    if (!desc_opt) {
        // 2. 创建描述符
        INFINICORE_CHECK_ERROR(infiniopCreatePairwiseDistanceDescriptor(
            context::getInfiniopHandle(output->device()), 
            &desc,
            output->desc(), 
            x1->desc(), 
            x2->desc(), 
            p,
            eps,
            keepdim // 直接传递 bool，C API 已经修改支持
        ));
        
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // 3. 获取 Workspace 并执行
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetPairwiseDistanceWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopPairwiseDistance(
        desc, 
        workspace->data(), 
        workspace_size,
        output->data(), 
        x1->data(), 
        x2->data(),
        context::getStream()
    ));
}

// 4. 注册算子实现到 Dispatcher
static bool registered = []() {
    PairwiseDistance::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::pairwise_distance_impl::infiniop