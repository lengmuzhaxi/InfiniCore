#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/margin_ranking_loss.h"

// --- 后端实现头文件 ---
#ifdef ENABLE_CPU_API
#include "cpu/margin_ranking_loss_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/margin_ranking_loss_nvidia.cuh"
#endif

#ifdef ENABLE_METAX_API
#include "metax/margin_ranking_loss_metax.h"
#endif

#ifdef ENABLE_MOORE_API
#include "moore/margin_ranking_loss_moore.h"
#endif

extern "C" {

// =======================================================================
// 1. 创建算子描述符
// =======================================================================
__C infiniStatus_t infiniopCreateMarginRankingLossDescriptor(
    infiniopHandle_t handle,
    infiniopMarginRankingLossDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t input1,
    infiniopTensorDescriptor_t input2,
    infiniopTensorDescriptor_t target,
    float margin,
    int p,            // <--- 新增参数
    int reduction) {

    #define CREATE(CASE, NAMESPACE)                                                             \
        case CASE:                                                                              \
            return op::margin_ranking_loss::NAMESPACE::Descriptor::create(                      \
                handle,                                                                         \
                reinterpret_cast<op::margin_ranking_loss::NAMESPACE::Descriptor **>(desc_ptr),  \
                output,                                                                         \
                input1,                                                                         \
                input2,                                                                         \
                target,                                                                         \
                margin,                                                                         \
                p,        /* <--- 传递 p 参数 */                                                 \
                reduction)

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
__C infiniStatus_t infiniopGetMarginRankingLossWorkspaceSize(infiniopMarginRankingLossDescriptor_t desc, size_t *size) {

    #define GET(CASE, NAMESPACE)                                                                    \
        case CASE:                                                                                  \
            *size = reinterpret_cast<op::margin_ranking_loss::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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
__C infiniStatus_t infiniopMarginRankingLoss(
    infiniopMarginRankingLossDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input1,
    const void *input2,
    const void *target,
    void *stream) {

    #define CALCULATE(CASE, NAMESPACE)                                                          \
        case CASE:                                                                              \
            return reinterpret_cast<const op::margin_ranking_loss::NAMESPACE::Descriptor *>(desc) \
                ->calculate(workspace, workspace_size, output, input1, input2, target, stream)

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
__C infiniStatus_t infiniopDestroyMarginRankingLossDescriptor(infiniopMarginRankingLossDescriptor_t desc) {

    #define DELETE(CASE, NAMESPACE)                                                             \
        case CASE:                                                                              \
            delete reinterpret_cast<const op::margin_ranking_loss::NAMESPACE::Descriptor *>(desc); \
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