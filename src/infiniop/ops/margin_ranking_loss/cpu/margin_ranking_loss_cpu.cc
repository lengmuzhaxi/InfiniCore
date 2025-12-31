#include "margin_ranking_loss_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>
#include <cstdint>
#include <numeric>

#include "../../../../utils/custom_types.h"

namespace op::margin_ranking_loss::cpu {

// ==================================================================
// 1. 定义公共数据结构 (用于 Opaque)
// ==================================================================
struct MarginRankingLossOpaqueData {
    int ndim;
    std::vector<size_t> output_shape; // 广播后的逻辑形状
    std::vector<int64_t> input1_strides;
    std::vector<int64_t> input2_strides;
    std::vector<int64_t> target_strides;
};

struct Descriptor::Opaque : public MarginRankingLossOpaqueData {};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
        _opaque = nullptr;
    }
}

// ==================================================================
// 2. 辅助函数：计算广播步长
// ==================================================================
static std::vector<int64_t> compute_broadcast_strides(
    const std::vector<size_t>& out_shape,
    infiniopTensorDescriptor_t input_desc) {
    
    int out_ndim = static_cast<int>(out_shape.size());
    const auto& in_shape = input_desc->shape();
    const auto& in_strides = input_desc->strides();
    int in_ndim = static_cast<int>(input_desc->ndim());
    
    std::vector<int64_t> effective_strides(out_ndim, 0);

    for (int i = 0; i < out_ndim; ++i) {
        int out_idx = out_ndim - 1 - i;
        int in_idx = in_ndim - 1 - i;

        if (in_idx >= 0) {
            size_t dim_size = in_shape[in_idx];
            // 如果输入维度为1（且对应输出维度>1），则该维度步长为0（广播）
            if (dim_size == 1) {
                effective_strides[out_idx] = 0; 
            } else {
                effective_strides[out_idx] = in_strides[in_idx];
            }
        } else {
            // 维度不足，自动广播
            effective_strides[out_idx] = 0; 
        }
    }
    return effective_strides;
}

// ==================================================================
// 3. Descriptor Creation
// ==================================================================
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input1_desc,
    infiniopTensorDescriptor_t input2_desc,
    infiniopTensorDescriptor_t target_desc,
    float margin,
    int p,
    int reduction) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    
    // 1. 创建 Info
    auto result = MarginRankingLossInfo::create(out_desc, input1_desc, input2_desc, target_desc, margin, p, reduction);
    CHECK_RESULT(result);
    auto info = result.take();

    // 2. 创建 Opaque 并推导广播形状
    // 注意：如果是 Reduction 模式，out_desc 是标量 (1,)，无法用于循环遍历。
    // 因此我们需要手动推导 Input1, Input2, Target 三者广播后的最大形状。
    auto opaque = new Opaque();

    // 推导最大秩
    int ndim = std::max({(int)input1_desc->ndim(), (int)input2_desc->ndim(), (int)target_desc->ndim()});
    std::vector<size_t> broadcast_shape(ndim, 1);

    // 推导每一维的大小
    for (int i = 0; i < ndim; ++i) {
        // 从右向左对齐
        size_t dim1 = (i < ndim - (int)input1_desc->ndim()) ? 1 : input1_desc->shape()[i - (ndim - input1_desc->ndim())];
        size_t dim2 = (i < ndim - (int)input2_desc->ndim()) ? 1 : input2_desc->shape()[i - (ndim - input2_desc->ndim())];
        size_t dim3 = (i < ndim - (int)target_desc->ndim()) ? 1 : target_desc->shape()[i - (ndim - target_desc->ndim())];
        
        broadcast_shape[i] = std::max({dim1, dim2, dim3});
    }
    
    opaque->ndim = ndim;
    opaque->output_shape = broadcast_shape;

    // 计算步长
    opaque->input1_strides = compute_broadcast_strides(opaque->output_shape, input1_desc);
    opaque->input2_strides = compute_broadcast_strides(opaque->output_shape, input2_desc);
    opaque->target_strides = compute_broadcast_strides(opaque->output_shape, target_desc);

    *desc_ptr = new Descriptor(
        opaque,
        info,
        0, // CPU 不需要 workspace
        handle->device, 
        handle->device_id
    );

    return INFINI_STATUS_SUCCESS;
}

// ==================================================================
// 4. Kernel Implementation
// ==================================================================
template <typename T>
void calculate_cpu_impl(
    const MarginRankingLossInfo &info,
    const MarginRankingLossOpaqueData *opaque,
    void *output,
    const void *input1,
    const void *input2,
    const void *target) {

    float margin = info.margin();
    int reduction = info.reduction();
    
    // 计算总元素数
    size_t numel = 1;
    for (auto s : opaque->output_shape) numel *= s;
    if (numel == 0) return;

    auto out_ptr = reinterpret_cast<T *>(output);
    auto in1_ptr = reinterpret_cast<const T *>(input1);
    auto in2_ptr = reinterpret_cast<const T *>(input2);
    auto tar_ptr = reinterpret_cast<const T *>(target);

    int ndim = opaque->ndim;
    const auto& shape = opaque->output_shape;
    const auto& str1 = opaque->input1_strides;
    const auto& str2 = opaque->input2_strides;
    const auto& str_tar = opaque->target_strides;

    // Element-wise 计算
    if (reduction == 0) { // None
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < numel; ++i) {
            size_t temp_idx = i;
            int64_t off1 = 0, off2 = 0, off_tar = 0;

            for (int d = ndim - 1; d >= 0; --d) {
                size_t coord = temp_idx % shape[d];
                temp_idx /= shape[d];
                off1 += coord * str1[d];
                off2 += coord * str2[d];
                off_tar += coord * str_tar[d];
            }

            float val1 = utils::cast<float>(in1_ptr[off1]);
            float val2 = utils::cast<float>(in2_ptr[off2]);
            float t = utils::cast<float>(tar_ptr[off_tar]);

            // Loss = max(0, -y * (x1 - x2) + margin)
            float val = -t * (val1 - val2) + margin;
            float loss = (val > 0.0f) ? val : 0.0f;

            out_ptr[i] = utils::cast<T>(loss);
        }
    } 
    // Reduction 计算 (Sum / Mean)
    else { 
        double total_loss = 0.0;

        #pragma omp parallel for reduction(+:total_loss) schedule(static)
        for (size_t i = 0; i < numel; ++i) {
            size_t temp_idx = i;
            int64_t off1 = 0, off2 = 0, off_tar = 0;

            for (int d = ndim - 1; d >= 0; --d) {
                size_t coord = temp_idx % shape[d];
                temp_idx /= shape[d];
                off1 += coord * str1[d];
                off2 += coord * str2[d];
                off_tar += coord * str_tar[d];
            }

            float val1 = utils::cast<float>(in1_ptr[off1]);
            float val2 = utils::cast<float>(in2_ptr[off2]);
            float t = utils::cast<float>(tar_ptr[off_tar]);

            float val = -t * (val1 - val2) + margin;
            float loss = (val > 0.0f) ? val : 0.0f;

            total_loss += static_cast<double>(loss);
        }

        if (reduction == 1) { // Mean
            total_loss /= static_cast<double>(numel);
        }
        
        out_ptr[0] = utils::cast<T>(static_cast<float>(total_loss));
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input1,
    const void *input2,
    const void *target,
    void *stream) const {

    auto dtype = _info.dtype();

    switch (dtype) {
    case INFINI_DTYPE_F32:
        cpu::calculate_cpu_impl<float>(_info, _opaque, output, input1, input2, target);
        break;
    case INFINI_DTYPE_F64:
        cpu::calculate_cpu_impl<double>(_info, _opaque, output, input1, input2, target);
        break;
    case INFINI_DTYPE_F16:
        cpu::calculate_cpu_impl<fp16_t>(_info, _opaque, output, input1, input2, target);
        break;
    case INFINI_DTYPE_BF16:
        cpu::calculate_cpu_impl<bf16_t>(_info, _opaque, output, input1, input2, target);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::margin_ranking_loss::cpu