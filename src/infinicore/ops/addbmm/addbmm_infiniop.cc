#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/addbmm.hpp" 
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>

namespace infinicore::op::addbmm_impl::infiniop {

// 1. 定义 Cache
thread_local common::OpCache<size_t, infiniopAddbmmDescriptor_t> caches(
    100, 
    [](infiniopAddbmmDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyAddbmmDescriptor(desc));
            desc = nullptr;
        }
    });

// 2. 实现 calculate 函数
// 【关键修复】将参数顺序改为 beta在前，alpha在后，与 Pybind 和 PyTorch 保持一致
void calculate(Tensor output, Tensor input, Tensor batch1, Tensor batch2, float beta, float alpha) {
    
    // Hash 也要对应修改顺序
    size_t seed = hash_combine(output, input, batch1, batch2, beta, alpha);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopAddbmmDescriptor_t desc = nullptr;
    
    if (!desc_opt) {
        // 创建描述符
        // 【注意】查看 InfiniOP 文档，确认 create 接口的参数顺序。
        // 通常 BLAS 风格是 (alpha, beta)。
        // 如果我们现在的变量 beta 存的是 0.5 (input系数), alpha 存的是 2.0 (matmul系数)
        // 那么我们要确保把正确的值传给 InfiniOP 对应的位置。
        INFINICORE_CHECK_ERROR(infiniopCreateAddbmmDescriptor(
            context::getInfiniopHandle(output->device()), 
            &desc,
            output->desc(), 
            input->desc(), 
            batch1->desc(), 
            batch2->desc(),
            alpha,  // 对应 InfiniOP 的 alpha (matmul系数)
            beta)); // 对应 InfiniOP 的 beta (input系数)
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // 获取 Workspace 大小
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetAddbmmWorkspaceSize(desc, &workspace_size));
    
    // 分配 Workspace
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    // 执行计算
    INFINICORE_CHECK_ERROR(infiniopAddbmm(
        desc, 
        workspace->data(), 
        workspace_size,
        output->data(), 
        input->data(), 
        batch1->data(), 
        batch2->data(),
        context::getStream()));
}

// 3. 注册算子实现
static bool registered = []() {
    Addbmm::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::addbmm_impl::infiniop