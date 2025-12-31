#include "pairwise_distance_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>

#include "../../../../utils/custom_types.h"

namespace op::pairwise_distance::cpu {

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
    infiniopTensorDescriptor_t x1_desc,
    infiniopTensorDescriptor_t x2_desc,
    float p,
    float eps,
    bool keepdim) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    
    // 创建 Info 对象
    // create 函数会校验形状兼容性 (N, D) 以及输出形状是否匹配 keepdim 设置
    auto result = PairwiseDistanceInfo::create(out_desc, x1_desc, x2_desc, p, eps, keepdim);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        new Opaque(),
        result.take(),
        0, 
        handle->device, 
        handle->device_id
    );

    return INFINI_STATUS_SUCCESS;
}

// 辅助函数：计算两个向量之间的 p-范数距离
template <typename T>
inline float compute_p_norm(const T* x, const T* y, size_t D, float p, float eps) {
    float sum = 0.0f;
    
    // 针对常见 p 值进行优化
    if (std::abs(p - 1.0f) < 1e-5f) { // p == 1 (Manhattan Distance)
        for (size_t i = 0; i < D; ++i) {
            sum += std::abs(utils::cast<float>(x[i]) - utils::cast<float>(y[i]));
        }
        return sum + eps; // 或者根据具体定义 return sum; eps 通常用于防止除零
    } 
    else if (std::abs(p - 2.0f) < 1e-5f) { // p == 2 (Euclidean Distance)
        for (size_t i = 0; i < D; ++i) {
            float diff = utils::cast<float>(x[i]) - utils::cast<float>(y[i]);
            sum += diff * diff;
        }
        return std::sqrt(std::max(0.0f, sum) + eps);
    } 
    else if (std::isinf(p)) { // p == inf (Chebyshev Distance)
         // 虽然通常 floating point 不直接判断 == inf，但为了完整性示意
         // 实际中可能通过 float p 特殊值传递
         for (size_t i = 0; i < D; ++i) {
            float diff = std::abs(utils::cast<float>(x[i]) - utils::cast<float>(y[i]));
            if (diff > sum) sum = diff;
        }
        return sum; // Infinity norm 不需要 eps 开方
    }
    else { // Generic p-norm
        for (size_t i = 0; i < D; ++i) {
            float diff = std::abs(utils::cast<float>(x[i]) - utils::cast<float>(y[i]));
            sum += std::pow(diff, p);
        }
        return std::pow(sum + eps, 1.0f / p);
    }
}

template <typename T>
void calculate_cpu_impl(
    const PairwiseDistanceInfo &info,
    void *output,
    const void *x1,
    const void *x2) {

    size_t N = info.batch_size();   // Batch Size
    size_t D = info.feature_dim();  // Feature Dimension
    float p = info.p();
    float eps = info.eps();
    // keepdim 参数影响输出形状校验，但不影响连续内存下的数据写入逻辑
    // (N) 和 (N, 1) 在内存中都是连续的 N 个元素

    auto out_ptr = reinterpret_cast<T *>(output);
    auto x1_ptr = reinterpret_cast<const T *>(x1);
    auto x2_ptr = reinterpret_cast<const T *>(x2);

    #pragma omp parallel for schedule(static)
    for (size_t n = 0; n < N; ++n) {
        // 定位到当前样本的起始位置
        const T* x1_row = x1_ptr + n * D;
        const T* x2_row = x2_ptr + n * D;

        // 计算距离
        float dist = compute_p_norm(x1_row, x2_row, D, p, eps);

        // 写入输出
        out_ptr[n] = utils::cast<T>(dist);
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *x1,
    const void *x2,
    void *stream) const {

    auto dtype = _info.dtype();

    switch (dtype) {
    case INFINI_DTYPE_F32:
        cpu::calculate_cpu_impl<float>(_info, output, x1, x2);
        break;
    case INFINI_DTYPE_F64:
        cpu::calculate_cpu_impl<double>(_info, output, x1, x2);
        break;
    case INFINI_DTYPE_F16:
        cpu::calculate_cpu_impl<fp16_t>(_info, output, x1, x2);
        break;
    case INFINI_DTYPE_BF16:
        cpu::calculate_cpu_impl<bf16_t>(_info, output, x1, x2);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::pairwise_distance::cpu