#include "hypot_moore.h"

// 引入 Moore 平台的通用 Elementwise 描述符宏
#include "../../../elementwise/moore/elementwise_moore.h"

// 引入 Hypot 的具体 Kernel 实现
#include "hypot_moore_kernel.h"

namespace op::hypot::moore {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    // Hypot is a binary operator (z = hypot(x, y) = sqrt(x^2 + y^2))
    // 需要确保有两个输入
    if (input_desc_vec.size() != 2) {
        return INFINI_STATUS_BAD_PARAM;
    }
    const auto &in_desc_0 = input_desc_vec.at(0);
    const auto &in_desc_1 = input_desc_vec.at(1);
    
    const auto &out_shape = out_desc->shape();
    const auto &in_shape_0 = in_desc_0->shape();
    const auto &in_shape_1 = in_desc_1->shape();

    // Hypot supports floating point types.
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    // Check if output shape matches input shapes
    // 这是一个 Elementwise 操作，通常要求形状一致 (或支持广播，视具体框架实现而定，此处模仿原代码保持严格一致)
    CHECK_SAME_SHAPE(out_shape, in_shape_0);
    CHECK_SAME_SHAPE(out_shape, in_shape_1);

    // create MOORE elementwise descriptor
    // 这里的宏会自动生成描述符初始化的通用代码
    CREATE_ELEMENTWISE_MOORE_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec)

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    
    // Safety check for input count in calculate phase
    if (inputs.size() != 2) {
        return INFINI_STATUS_BAD_PARAM;
    }

    // Use moore::HypotOp template defined in hypot_moore_kernel.h
    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return _device_info->calculate<256, moore::HypotOp, half>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<256, moore::HypotOp, cuda_bfloat16>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<256, moore::HypotOp, float>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<256, moore::HypotOp, double>(_info, workspace, output, inputs, stream);
    
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::hypot::moore