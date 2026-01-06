#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/split.h"

// --- 后端实现头文件 ---
#ifdef ENABLE_CPU_API
#include "cpu/split_cpu.h"
#endif

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/split_nvidia.cuh"
#endif

#ifdef ENABLE_METAX_API
#include "metax/split_metax.h"
#endif

#ifdef ENABLE_MOORE_API
#include "moore/split_moore.h"
#endif

extern "C" {

// =======================================================================
// 1. 创建算子描述符
// =======================================================================
__C infiniStatus_t infiniopCreateSplitDescriptor(
    infiniopHandle_t handle,
    infiniopSplitDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t const *ys, 
    uint64_t num_outputs,
    infiniopTensorDescriptor_t x,
    int64_t axis) {

    // 将 C 风格的输出描述符数组转换为 std::vector
    std::vector<infiniopTensorDescriptor_t> output_descs(ys, ys + num_outputs);

    #define CREATE(CASE, NAMESPACE)                                             \
        case CASE:                                                              \
            return op::split::NAMESPACE::Descriptor::create(                    \
                handle,                                                         \
                reinterpret_cast<op::split::NAMESPACE::Descriptor **>(desc_ptr),\
                output_descs,                                                   \
                {x}, /* 单个输入转 vector */                                     \
                axis)

    switch (handle->device) {
    #ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
    #endif
    #ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
    #endif
    #ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
    #endif
    #ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia);
    #endif
    #ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
    #endif
    #ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, moore);
    #endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
    #undef CREATE
}

// =======================================================================
// 2. 获取 Workspace 大小
// =======================================================================
__C infiniStatus_t infiniopGetSplitWorkspaceSize(infiniopSplitDescriptor_t desc, size_t *size) {

    #define GET(CASE, NAMESPACE)                                                                    \
        case CASE:                                                                                  \
            *size = reinterpret_cast<op::split::NAMESPACE::Descriptor *>(desc)->workspaceSize();    \
            return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
    #ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
    #endif
    #ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
    #endif
    #ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
    #endif
    #ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia);
    #endif
    #ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
    #endif
    #ifdef ENABLE_MOORE_API
        GET(INFINI_DEVICE_MOORE, moore);
    #endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
    #undef GET
}

// =======================================================================
// 3. 执行计算 (Calculate)
// =======================================================================
__C infiniStatus_t infiniopSplit(
    infiniopSplitDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *const *ys,
    const void *x,
    void *stream) {

    #define CALCULATE(CASE, NAMESPACE)                                                          \
        case CASE: {                                                                            \
            auto *d = reinterpret_cast<const op::split::NAMESPACE::Descriptor *>(desc);         \
            /* 修正点：调用 numOutputs() 方法 */                                                 \
            std::vector<void *> output_ptrs(ys, ys + d->numOutputs());                          \
            return d->calculate(workspace, workspace_size, output_ptrs, {x}, stream);           \
        }

    switch (desc->device_type) {
    #ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
    #endif
    #ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
    #endif
    #ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
    #endif
    #ifdef ENABLE_QY_API
        CALCULATE(INFINI_DEVICE_QY, nvidia);
    #endif
    #ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
    #endif
    #ifdef ENABLE_MOORE_API
        CALCULATE(INFINI_DEVICE_MOORE, moore);
    #endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
    #undef CALCULATE
}

// =======================================================================
// 4. 销毁描述符
// =======================================================================
__C infiniStatus_t infiniopDestroySplitDescriptor(infiniopSplitDescriptor_t desc) {

    #define DELETE(CASE, NAMESPACE)                                                              \
        case CASE:                                                                               \
            delete reinterpret_cast<const op::split::NAMESPACE::Descriptor *>(desc);             \
            return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
    #ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
    #endif
    #ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
    #endif
    #ifdef ENABLE_ILUVATAR_API
        DELETE(INFINI_DEVICE_ILUVATAR, nvidia);
    #endif
    #ifdef ENABLE_QY_API
        DELETE(INFINI_DEVICE_QY, nvidia);
    #endif
    #ifdef ENABLE_METAX_API
        DELETE(INFINI_DEVICE_METAX, metax);
    #endif
    #ifdef ENABLE_MOORE_API
        DELETE(INFINI_DEVICE_MOORE, moore);
    #endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
    #undef DELETE
}

} // extern "C"