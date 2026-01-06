#include "softshrink_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <vector>
#include <cmath>
#include <omp.h>
#include <type_traits>

#include "../../../../utils/custom_types.h"

namespace op::softshrink::cpu {

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
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec,
    float lambd) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    if (input_desc_vec.empty()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // 调用 SoftshrinkInfo::create
    auto result = SoftshrinkInfo::create(out_desc, input_desc_vec[0], lambd);
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
// 核心修改：支持 Strided Memory (非连续内存)
// ==================================================================
template <typename T>
void calculate_cpu_impl(
    const SoftshrinkInfo &info,
    void *output,
    const void *input) {

    size_t total_elements = info.total_elements();
    float lambd = info.lambd();

    // 获取形状和步长信息 (之前在 info.h 中添加的)
    int ndim = info.ndim();
    const size_t* shape = info.shape();
    const ptrdiff_t* strides = info.strides();

    auto out_ptr = reinterpret_cast<T *>(output);
    auto in_ptr = reinterpret_cast<const T *>(input);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total_elements; ++i) {
        // --- 1. 计算输入的物理偏移量 (Offset Calculation) ---
        size_t input_offset = 0;
        size_t remaining = i;
        
        // 将线性索引 i 转换为多维坐标，并结合 stride 计算 offset
        for (int d = ndim - 1; d >= 0; --d) {
            size_t dim_idx = remaining % shape[d];
            remaining /= shape[d];
            input_offset += dim_idx * strides[d];
        }

        // --- 2. 读取数据 (使用计算出的偏移量) ---
        T x = in_ptr[input_offset];
        
        // --- 3. 计算逻辑 (保持不变) ---
        if constexpr (std::is_arithmetic_v<T>) {
            if (x > static_cast<T>(lambd)) {
                out_ptr[i] = x - static_cast<T>(lambd);
            } else if (x < -static_cast<T>(lambd)) {
                out_ptr[i] = x + static_cast<T>(lambd);
            } else {
                out_ptr[i] = static_cast<T>(0);
            }
        } else {
            // 自定义类型 (fp16/bf16) 使用 utils::cast
            float xf = utils::cast<float>(x);
            float res = 0.0f;
            if (xf > lambd) {
                res = xf - lambd;
            } else if (xf < -lambd) {
                res = xf + lambd;
            } else {
                res = 0.0f;
            }
            out_ptr[i] = utils::cast<T>(res);
        }
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    if (inputs.empty()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    const void *input = inputs[0];
    auto dtype = _info.dtype();

    switch (dtype) {
    case INFINI_DTYPE_F32:
        cpu::calculate_cpu_impl<float>(_info, output, input);
        break;
    case INFINI_DTYPE_F64:
        cpu::calculate_cpu_impl<double>(_info, output, input);
        break;
    case INFINI_DTYPE_F16:
        cpu::calculate_cpu_impl<fp16_t>(_info, output, input);
        break;
    case INFINI_DTYPE_BF16:
        cpu::calculate_cpu_impl<bf16_t>(_info, output, input);
        break;
    case INFINI_DTYPE_I32:
        cpu::calculate_cpu_impl<int32_t>(_info, output, input);
        break;
    case INFINI_DTYPE_I64:
        cpu::calculate_cpu_impl<int64_t>(_info, output, input);
        break;
    case INFINI_DTYPE_U32:
        cpu::calculate_cpu_impl<uint32_t>(_info, output, input);
        break;
    case INFINI_DTYPE_U64:
        cpu::calculate_cpu_impl<uint64_t>(_info, output, input);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::softshrink::cpu