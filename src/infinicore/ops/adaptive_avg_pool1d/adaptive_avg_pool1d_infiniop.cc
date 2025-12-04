#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/adaptive_avg_pool1d.hpp" // 引用算子定义
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>

namespace infinicore::op::adaptive_avg_pool1d_impl::infiniop {

// 定义线程局部缓存，用于存储算子描述符
// Key: size_t (Hash)
// Value: infiniopAdaptiveAvgPool1dDescriptor_t
thread_local common::OpCache<size_t, infiniopAdaptiveAvgPool1dDescriptor_t> caches(
    100, // capacity
    [](infiniopAdaptiveAvgPool1dDescriptor_t &desc) {
        if (desc != nullptr) {
            // 销毁描述符
            INFINICORE_CHECK_ERROR(infiniopDestroyAdaptiveAvgPool1dDescriptor(desc));
            desc = nullptr;
        }
    });

// 计算函数实现
void calculate(Tensor output, Tensor input) {
    // 1. 计算 Hash Key
    // output 中已经包含了 output_size 的形状信息，因此 hash(output, input) 就足够唯一标识
    size_t seed = hash_combine(output, input);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    // 获取当前设备对应的缓存
    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopAdaptiveAvgPool1dDescriptor_t desc = nullptr;

    // 2. 缓存未命中，创建描述符
    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateAdaptiveAvgPool1dDescriptor(
            context::getInfiniopHandle(output->device()), &desc,
            output->desc(), input->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // 3. 获取 Workspace 大小并分配
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetAdaptiveAvgPool1dWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    // 4. 执行计算
    INFINICORE_CHECK_ERROR(infiniopAdaptiveAvgPool1d(
        desc, 
        workspace->data(), 
        workspace_size,
        output->data(), 
        input->data(), 
        context::getStream()));
}

// 5. 注册算子到 Dispatcher
static bool registered = []() {
    AdaptiveAvgPool1d::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::adaptive_avg_pool1d_impl::infiniop