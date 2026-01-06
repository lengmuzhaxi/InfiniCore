#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/split.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>
#include <unordered_map>
#include <vector>

namespace infinicore::op::split_impl::infiniop {

thread_local common::OpCache<size_t, infiniopSplitDescriptor_t> caches(
    100,
    [](infiniopSplitDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroySplitDescriptor(desc));
            desc = nullptr;
        }
    }
);

struct WorkspaceEntry {
    size_t size = 0;
    std::shared_ptr<Memory> buf = nullptr;
};

void calculate(std::vector<Tensor> outputs, Tensor input, int64_t dim) {
    // 【修复 1】hash_combine 返回 void，需初始化 seed 并通过引用传递修改
    size_t seed = 0;
    hash_combine(seed, input);
    hash_combine(seed, dim);
    for (const auto &out : outputs) {
        hash_combine(seed, out);
    }

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopSplitDescriptor_t desc = nullptr;
    if (!desc_opt) {
        std::vector<infiniopTensorDescriptor_t> output_descs;
        output_descs.reserve(outputs.size());
        for (const auto &out : outputs) {
            output_descs.push_back(out->desc());
        }

        INFINICORE_CHECK_ERROR(infiniopCreateSplitDescriptor(
            context::getInfiniopHandle(input->device()), &desc,
            output_descs.data(), output_descs.size(),
            input->desc(), dim));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    static thread_local std::unordered_map<infiniopSplitDescriptor_t, WorkspaceEntry> s_workspace_map;
    auto it = s_workspace_map.find(desc);

    if (it == s_workspace_map.end()) {
        size_t workspace_size = 0;
        INFINICORE_CHECK_ERROR(infiniopGetSplitWorkspaceSize(desc, &workspace_size));

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
        INFINICORE_CHECK_ERROR(infiniopGetSplitWorkspaceSize(desc, &required_size));
        if (required_size > it->second.size) {
            it->second.buf = context::allocateMemory(required_size);
            it->second.size = required_size;
        }
    }

    void *workspace_ptr = (it != s_workspace_map.end() && it->second.buf) ? it->second.buf->data() : nullptr;
    size_t workspace_size = (it != s_workspace_map.end()) ? it->second.size : 0;

    std::vector<void *> output_ptrs;
    output_ptrs.reserve(outputs.size());
    for (const auto &out : outputs) {
        // 【修复 2】out->data() 返回 const 指针，需要 const_cast 转为 void* 以便写入
        output_ptrs.push_back(const_cast<void*>(static_cast<const void*>(out->data())));
    }

    INFINICORE_CHECK_ERROR(infiniopSplit(
        desc,
        workspace_ptr,
        workspace_size,
        output_ptrs.data(),
        input->data(),
        context::getStream()));
}

static bool registered = []() {
    Split::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::split_impl::infiniop