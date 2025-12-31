#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/pairwise_distance.h"

// --- 后端实现头文件 ---
#ifdef ENABLE_CPU_API
#include "cpu/pairwise_distance_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/pairwise_distance_nvidia.cuh"
#endif

#ifdef ENABLE_METAX_API
#include "metax/pairwise_distance_metax.h"
#endif

#ifdef ENABLE_MOORE_API
#include "moore/pairwise_distance_moore.h"
#endif

extern "C" {

// =======================================================================
// 1. 创建算子描述符
// =======================================================================
__C infiniStatus_t infiniopCreatePairwiseDistanceDescriptor(
    infiniopHandle_t handle,
    infiniopPairwiseDistanceDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t x1,
    infiniopTensorDescriptor_t x2,
    float p,
    float eps,
    bool keepdim) {

    #define CREATE(CASE, NAMESPACE)                                                             \
        case CASE:                                                                              \
            return op::pairwise_distance::NAMESPACE::Descriptor::create(                        \
                handle,                                                                         \
                reinterpret_cast<op::pairwise_distance::NAMESPACE::Descriptor **>(desc_ptr),    \
                output,                                                                         \
                x1,                                                                             \
                x2,                                                                             \
                p,                                                                              \
                eps,                                                                            \
                keepdim)

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
__C infiniStatus_t infiniopGetPairwiseDistanceWorkspaceSize(infiniopPairwiseDistanceDescriptor_t desc, size_t *size) {

    #define GET(CASE, NAMESPACE)                                                                                 \
        case CASE:                                                                                               \
            *size = reinterpret_cast<op::pairwise_distance::NAMESPACE::Descriptor *>(desc)->workspaceSize();     \
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
__C infiniStatus_t infiniopPairwiseDistance(
    infiniopPairwiseDistanceDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *x1,
    const void *x2,
    void *stream) {

    #define CALCULATE(CASE, NAMESPACE)                                                            \
        case CASE:                                                                                \
            return reinterpret_cast<const op::pairwise_distance::NAMESPACE::Descriptor *>(desc)   \
                ->calculate(workspace, workspace_size, output, x1, x2, stream)

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
// 4. 销毁算子描述符
// =======================================================================
__C infiniStatus_t infiniopDestroyPairwiseDistanceDescriptor(infiniopPairwiseDistanceDescriptor_t desc) {

    #define DELETE(CASE, NAMESPACE)                                                                         \
        case CASE:                                                                                          \
            delete reinterpret_cast<const op::pairwise_distance::NAMESPACE::Descriptor *>(desc);            \
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