#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/softshrink.hpp" // 引入 Softshrink 头文件
#include <infiniop.h>

namespace infinicore::op::softshrink_impl::infiniop {

// 定义描述符缓存，类型为 infiniopSoftshrinkDescriptor_t
thread_local common::OpCache<size_t, infiniopSoftshrinkDescriptor_t> caches(
    100, // capacity
    [](infiniopSoftshrinkDescriptor_t &desc) {
        if (desc != nullptr) {
            // 销毁 Softshrink 描述符
            INFINICORE_CHECK_ERROR(infiniopDestroySoftshrinkDescriptor(desc));
            desc = nullptr;
        }
    });

// 计算函数实现
void calculate(Tensor output, Tensor input, float lambd) {
    size_t seed = hash_combine(output, input, lambd);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopSoftshrinkDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateSoftshrinkDescriptor(
            context::getInfiniopHandle(output->device()), &desc,
            output->desc(), input->desc(), lambd));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetSoftshrinkWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopSoftshrink(
        desc,
        workspace->data(), workspace_size,
        output->data(), input->data(), // 参数顺序通常是 Output, Input
        context::getStream()));
}

static bool registered = []() {
    Softshrink::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::softshrink_impl::infiniop
