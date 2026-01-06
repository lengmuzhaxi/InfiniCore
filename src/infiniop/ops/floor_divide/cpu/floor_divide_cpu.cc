#include "floor_divide_cpu.h"
#include <vector>
#include <cstring> // 用于 std::memcpy

namespace op::floor_divide::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &a_desc = input_desc_vec.at(0);
    const auto &b_desc = input_desc_vec.at(1);
    const auto &c_shape = out_desc->shape();
    const auto &a_shape = a_desc->shape();
    const auto &b_shape = b_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16, INFINI_DTYPE_I32, INFINI_DTYPE_I64);

    CHECK_SAME_SHAPE(c_shape, a_shape, b_shape);

    // create CPU elementwise descriptor
    CREATE_ELEMENTWISE_CPU_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec);

    return INFINI_STATUS_SUCCESS;
}

// ----------------------------------------------------------------------
// 辅助函数：安全计算逻辑
// ----------------------------------------------------------------------
// 目的：处理 In-place (out == in_1) 的情况。
// 当结果写回的位置恰好是除数的位置时，会破坏尚未计算的数据。
// 此函数检测冲突并自动建立除数的临时副本。
// ----------------------------------------------------------------------
template <typename T, typename DeviceInfo, typename Info>
infiniStatus_t SafeCalculate(
    DeviceInfo* device_info,
    const Info& info,
    void* output,
    const std::vector<const void*>& inputs,
    void* stream) {

    // 检查：输出地址是否与除数(inputs[1])地址相同
    if (output == inputs[1]) {
        // 获取元素总数
        // [修复] 根据 elementwise.h，使用 getOutputSize()
        size_t num_elements = info.getOutputSize(); 

        // 1. 分配临时内存
        std::vector<T> temp_buffer(num_elements);
        
        // 2. 将原始除数数据拷贝到临时内存
        // 注意：这里假设 In-place 时内存是连续的或者布局一致的
        // 如果输入可能有 stride 导致的不连续，memcpy 拷贝的是“物理上的连续内存块”
        // 只要 num_elements * sizeof(T) 覆盖了所需读取的范围即可。
        // 但对于 Elementwise 算子，In-place 通常意味着 shape/stride 完全一致。
        const T* src_ptr = static_cast<const T*>(inputs[1]);
        std::memcpy(temp_buffer.data(), src_ptr, num_elements * sizeof(T));

        // 3. 构建新的输入列表，将除数指针指向临时内存
        std::vector<const void*> new_inputs = inputs;
        new_inputs[1] = temp_buffer.data();

        // 4. 调用底层的 calculate，使用安全的 new_inputs
        // calculate 执行完毕后，temp_buffer 会自动析构，释放内存
        return device_info->template calculate<FloorDivideOp, T>(info, output, new_inputs, stream);
    }

    // 如果没有地址冲突，直接执行原逻辑，零开销
    return device_info->template calculate<FloorDivideOp, T>(info, output, inputs, stream);
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        // [修复] 增加 .get()，从 unique_ptr 获取原始指针
        return SafeCalculate<fp16_t>(_device_info.get(), _info, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return SafeCalculate<float>(_device_info.get(), _info, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return SafeCalculate<double>(_device_info.get(), _info, output, inputs, stream);
    case INFINI_DTYPE_BF16:
        return SafeCalculate<bf16_t>(_device_info.get(), _info, output, inputs, stream);
        
    case INFINI_DTYPE_I32:
        return SafeCalculate<int32_t>(_device_info.get(), _info, output, inputs, stream);
    case INFINI_DTYPE_I64:
        return SafeCalculate<int64_t>(_device_info.get(), _info, output, inputs, stream);
        
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::floor_divide::cpu