#include "split_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <vector>
#include <cmath>
#include <omp.h>
#include <type_traits>

#include "../../../../utils/custom_types.h"

namespace op::split::cpu {

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
        _opaque = nullptr;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    std::vector<infiniopTensorDescriptor_t> output_descs,
    std::vector<infiniopTensorDescriptor_t> input_descs,
    int64_t axis) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    if (input_descs.empty() || output_descs.empty()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // 调用 SplitInfo::create
    auto result = SplitInfo::create(output_descs, input_descs[0], axis);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        new Opaque(),
        result.take(),
        0, // workspace size
        handle->device, 
        handle->device_id
    );

    return INFINI_STATUS_SUCCESS;
}

// ==================================================================
// 核心修改：支持 Strided Memory 的 Split 逻辑
// ==================================================================
template <typename T>
void calculate_cpu_impl(
    const SplitInfo &info,
    std::vector<void *> outputs,
    const void *input) {

    // 获取输入的基本信息
    int ndim = info.ndim();
    int64_t axis = info.axis();
    
    // 【修复】删除未使用的 in_shape 变量
    // const size_t* in_shape = info.input_shape();
    
    const ptrdiff_t* in_strides = info.input_strides();

    auto in_ptr = reinterpret_cast<const T *>(input);
    const auto& output_metas = info.outputs();

    // 记录在 axis 维度上的当前偏移量 (以元素个数为单位)
    size_t current_axis_offset = 0;

    // --- 外层循环：遍历每一个输出 Tensor ---
    for (size_t k = 0; k < outputs.size(); ++k) {
        auto out_ptr = reinterpret_cast<T *>(outputs[k]);
        const auto& meta = output_metas[k];
        
        // 计算当前输出 Tensor 的总元素数量
        size_t out_total_elements = 1;
        for (int d = 0; d < ndim; ++d) {
            out_total_elements *= meta.shape[d];
        }

        // --- 内层循环：并行拷贝当前分片的数据 ---
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < out_total_elements; ++i) {
            // 1. 坐标映射计算
            size_t remaining = i;
            size_t output_physical_offset = 0;
            size_t input_physical_offset = 0;

            // 从后向前遍历维度，将线性索引 i 转换为多维坐标
            for (int d = ndim - 1; d >= 0; --d) {
                // 当前维度的坐标索引
                size_t dim_idx = remaining % meta.shape[d];
                remaining /= meta.shape[d];

                // A. 计算 Output 的物理偏移
                output_physical_offset += dim_idx * meta.strides[d];

                // B. 计算 Input 的物理偏移
                // 如果是 Split 轴 (axis)，需要加上当前分片的起始偏移量
                if (d == axis) {
                    input_physical_offset += (dim_idx + current_axis_offset) * in_strides[d];
                } else {
                    input_physical_offset += dim_idx * in_strides[d];
                }
            }

            // 2. 执行拷贝
            out_ptr[output_physical_offset] = in_ptr[input_physical_offset];
        }

        // 更新 axis 偏移量，为下一个输出 Tensor 做准备
        current_axis_offset += meta.shape[axis];
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    std::vector<void *> outputs,
    std::vector<const void *> inputs,
    void *stream) const {

    if (inputs.empty() || outputs.empty()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    const void *input = inputs[0];
    auto dtype = _info.dtype();

    // 检查输出数量是否匹配
    if (outputs.size() != _info.outputs().size()) {
        return INFINI_STATUS_BAD_PARAM;
    }

    switch (dtype) {
    case INFINI_DTYPE_F32:
        cpu::calculate_cpu_impl<float>(_info, outputs, input);
        break;
    case INFINI_DTYPE_F64:
        cpu::calculate_cpu_impl<double>(_info, outputs, input);
        break;
    case INFINI_DTYPE_F16:
        cpu::calculate_cpu_impl<fp16_t>(_info, outputs, input);
        break;
    case INFINI_DTYPE_BF16:
        cpu::calculate_cpu_impl<bf16_t>(_info, outputs, input);
        break;
    case INFINI_DTYPE_I32:
        cpu::calculate_cpu_impl<int32_t>(_info, outputs, input);
        break;
    case INFINI_DTYPE_I64:
        cpu::calculate_cpu_impl<int64_t>(_info, outputs, input);
        break;
    case INFINI_DTYPE_U32:
        cpu::calculate_cpu_impl<uint32_t>(_info, outputs, input);
        break;
    case INFINI_DTYPE_U64:
        cpu::calculate_cpu_impl<uint64_t>(_info, outputs, input);
        break;
    case INFINI_DTYPE_I8:
        cpu::calculate_cpu_impl<int8_t>(_info, outputs, input);
        break;
    case INFINI_DTYPE_U8:
        cpu::calculate_cpu_impl<uint8_t>(_info, outputs, input);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::split::cpu