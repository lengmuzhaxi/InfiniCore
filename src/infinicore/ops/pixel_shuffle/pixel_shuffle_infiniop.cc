#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/pixel_shuffle.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>
#include <vector>

namespace infinicore::op::pixel_shuffle_impl::infiniop {

// -------------------------------------------
// 1. 定义资源上下文
// -------------------------------------------
struct PixelShuffleContext {
    infiniopPixelShuffleDescriptor_t desc = nullptr;
    std::shared_ptr<Memory> workspace_buf = nullptr;
    size_t workspace_size = 0;

    void* getWorkspacePtr() const {
        return workspace_buf ? workspace_buf->data() : nullptr;
    }
};

// -------------------------------------------
// 2. 统一 LRU 缓存
// -------------------------------------------
thread_local common::OpCache<size_t, PixelShuffleContext> caches(
    256,
    [](PixelShuffleContext &ctx) {
        if (ctx.desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyPixelShuffleDescriptor(ctx.desc));
            ctx.desc = nullptr;
        }
        ctx.workspace_buf = nullptr;
    }
);

// -------------------------------------------
// 3. 计算 Hash
// -------------------------------------------
inline size_t compute_key(const Tensor& output, const Tensor& input, int upscale_factor) {
    size_t seed = 0;
    // 指针哈希
    infinicore::hash_combine(seed, reinterpret_cast<size_t>(output.operator->()));
    infinicore::hash_combine(seed, reinterpret_cast<size_t>(input.operator->()));
    // 标量哈希
    infinicore::hash_combine(seed, upscale_factor);
    return seed;
}

// -------------------------------------------
// 4. 核心计算函数
// -------------------------------------------
void calculate(Tensor output, Tensor input, int upscale_factor) {
    
    // 1. 计算 Hash
    size_t seed = compute_key(output, input, upscale_factor);

    // 2. 极速路径 (Fast Path) 变量
    static thread_local size_t last_seed = 0;
    static thread_local bool last_ctx_valid = false;
    static thread_local PixelShuffleContext last_ctx;

    PixelShuffleContext* ctx_ptr = nullptr;

    // 3. 检查 Fast Path
    if (last_ctx_valid && seed == last_seed) {
        ctx_ptr = &last_ctx;
    } else {
        // 4. 慢路径：查 LRU Cache
        auto device_type = context::getDevice().getType();
        auto device_index = context::getDevice().getIndex();
        auto &cache = caches.getCache(device_type, device_index);

        auto opt_ctx = cache.get(seed);
        if (opt_ctx) {
            last_ctx = *opt_ctx;
        } else {
            // 未命中：创建资源
            PixelShuffleContext new_ctx;
            
            // A. 创建 Descriptor
            INFINICORE_CHECK_ERROR(infiniopCreatePixelShuffleDescriptor(
                context::getInfiniopHandle(output->device()), 
                &new_ctx.desc,
                output->desc(), 
                input->desc(),
                upscale_factor));

            // B. 获取并分配 Workspace
            INFINICORE_CHECK_ERROR(infiniopGetPixelShuffleWorkspaceSize(new_ctx.desc, &new_ctx.workspace_size));
            
            if (new_ctx.workspace_size > 0) {
                new_ctx.workspace_buf = context::allocateMemory(new_ctx.workspace_size);
            }

            // C. 存入缓存
            cache.put(seed, new_ctx);
            last_ctx = new_ctx;
        }

        // 更新 Fast Path
        last_seed = seed;
        last_ctx_valid = true;
        ctx_ptr = &last_ctx;
    }

    // 5. 执行计算
    INFINICORE_CHECK_ERROR(infiniopPixelShuffle(
        ctx_ptr->desc, 
        ctx_ptr->getWorkspacePtr(), 
        ctx_ptr->workspace_size,
        output->data(), 
        input->data(), 
        context::getStream()));
}

// -------------------------------------------
// 5. 注册算子
// -------------------------------------------
static bool registered = []() {
    PixelShuffle::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::pixel_shuffle_impl::infiniop