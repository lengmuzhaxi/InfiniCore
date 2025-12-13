#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/matrix_power.h"

// --- 后端头文件 ---
#ifdef ENABLE_CPU_API
#include "cpu/matrix_power_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/matrix_power_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/matrix_power_metax.h"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/matrix_power_moore.h"
#endif

extern "C" {

// =======================================================================
// 1. 创建算子描述符
// =======================================================================
// 修正：移除 int64_t n，匹配头文件声明
__C infiniStatus_t infiniopCreateMatrixPowerDescriptor(
    infiniopHandle_t handle,
    infiniopMatrixPowerDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t input) {

    #define CREATE(CASE, NAMESPACE)                                             \
        case CASE:                                                              \
            return op::matrix_power::NAMESPACE::Descriptor::create(             \
                handle,                                                         \
                reinterpret_cast<op::matrix_power::NAMESPACE::Descriptor **>(desc_ptr), \
                output,                                                         \
                input) 
                // 注意：这里不再传递 n，因为 n 是 calculate 时的参数
                // 后端 Descriptor::create 实现也需要相应修改为不接收 n

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
__C infiniStatus_t infiniopGetMatrixPowerWorkspaceSize(infiniopMatrixPowerDescriptor_t desc, size_t *size) {

    #define GET(CASE, NAMESPACE)                                                                    \
        case CASE:                                                                                  \
            *size = reinterpret_cast<op::matrix_power::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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
// 修正：添加 int n，匹配头文件声明
__C infiniStatus_t infiniopMatrixPower(
    infiniopMatrixPowerDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    int n, // [修正] 这里的 n 是运行时参数
    void *stream) {

    #define CALCULATE(CASE, NAMESPACE)                                          \
        case CASE:                                                              \
            return reinterpret_cast<const op::matrix_power::NAMESPACE::Descriptor *>(desc) \
                ->calculate(workspace, workspace_size, output, input, n, stream)
                // 注意：这里将 n 传递给后端的 calculate

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
__C infiniStatus_t infiniopDestroyMatrixPowerDescriptor(infiniopMatrixPowerDescriptor_t desc) {

    #define DELETE(CASE, NAMESPACE)                                                             \
        case CASE:                                                                              \
            delete reinterpret_cast<const op::matrix_power::NAMESPACE::Descriptor *>(desc);     \
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