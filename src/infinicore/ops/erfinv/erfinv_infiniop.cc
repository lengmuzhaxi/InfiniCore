#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/erfinv.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>
#include <unordered_map>

namespace infinicore::op::erfinv_impl::infiniop {

thread_local common::OpCache<size_t, infiniopErfinvDescriptor_t> caches(
    100,
    [](infiniopErfinvDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyErfinvDescriptor(desc));
            desc = nullptr;
        }
    }
);

struct WorkspaceEntry {
    size_t size = 0;
    std::shared_ptr<Memory> buf = nullptr;
};

void calculate(Tensor output, Tensor input) {
    size_t seed = hash_combine(output, input);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopErfinvDescriptor_t desc = nullptr;
    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateErfinvDescriptor(
            context::getInfiniopHandle(output->device()), &desc,
            output->desc(), input->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    static thread_local std::unordered_map<infiniopErfinvDescriptor_t, WorkspaceEntry> s_workspace_map;
    auto it = s_workspace_map.find(desc);

    if (it == s_workspace_map.end()) {
        size_t workspace_size = 0;
        INFINICORE_CHECK_ERROR(infiniopGetErfinvWorkspaceSize(desc, &workspace_size));

        WorkspaceEntry entry;
        if (workspace_size > 0) {
            entry.buf = context::allocateMemory(workspace_size);
            entry.size = workspace_size;
        } else {
            entry.buf = nullptr;
            entry.size = 0;
        }
        it = s_workspace_map.emplace(desc, std::move(entry)).first;
    } else {
        size_t required_size = 0;
        INFINICORE_CHECK_ERROR(infiniopGetErfinvWorkspaceSize(desc, &required_size));
        if (required_size > it->second.size) {
            it->second.buf = context::allocateMemory(required_size);
            it->second.size = required_size;
        }
    }

    void* workspace_ptr = (it != s_workspace_map.end() && it->second.buf) ? it->second.buf->data() : nullptr;
    size_t workspace_size = (it != s_workspace_map.end()) ? it->second.size : 0;

    INFINICORE_CHECK_ERROR(infiniopErfinv(
        desc,
        workspace_ptr,
        workspace_size,
        output->data(),
        input->data(),
        context::getStream()
    ));
}

static bool registered = []() {
    Erfinv::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::erfinv_impl::infiniop