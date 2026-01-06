#include "triplet_margin_loss_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>

#include "../../../../utils/custom_types.h"

namespace op::triplet_margin_loss::cpu {

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
    infiniopTensorDescriptor_t anchor_desc,
    infiniopTensorDescriptor_t positive_desc,
    infiniopTensorDescriptor_t negative_desc,
    float margin,
    int p,
    float eps,
    int swap,
    int reduction) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    
    auto result = TripletMarginLossInfo::create(out_desc, anchor_desc, positive_desc, negative_desc, margin, p, eps, swap, reduction);
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

// 【终极修正】：完全对齐 PyTorch CPU 端 Forward 行为，不加 eps
template <typename T>
inline double compute_distance_stable(const T* x, const T* y, size_t D, int p) {
    double sum = 0.0;
    
    if (p == 1) {
        for (size_t i = 0; i < D; ++i) {
            double diff = std::abs(utils::cast<double>(x[i]) - utils::cast<double>(y[i]));
            sum += diff;
        }
        return sum; 
    } 
    else if (p == 2) {
        for (size_t i = 0; i < D; ++i) {
            double diff = utils::cast<double>(x[i]) - utils::cast<double>(y[i]);
            sum += diff * diff;
        }
        return std::sqrt(sum);
    } 
    else {
        double p_d = static_cast<double>(p);
        for (size_t i = 0; i < D; ++i) {
            double diff = std::abs(utils::cast<double>(x[i]) - utils::cast<double>(y[i]));
            sum += std::pow(diff, p_d);
        }
        return std::pow(sum, 1.0 / p_d);
    }
}

template <typename T>
void calculate_cpu_impl(
    const TripletMarginLossInfo &info,
    void *output,
    const void *anchor,
    const void *positive,
    const void *negative) {

    size_t N = info.batch_size();
    size_t D = info.feature_dim();
    double margin = static_cast<double>(info.margin());
    int p = info.p();
    bool swap = info.swap();
    int reduction = info.reduction();

    auto out_ptr = reinterpret_cast<T *>(output);
    auto anc_ptr = reinterpret_cast<const T *>(anchor);
    auto pos_ptr = reinterpret_cast<const T *>(positive);
    auto neg_ptr = reinterpret_cast<const T *>(negative);

    // 特判 N=1
    if (N == 1) {
        double d_p = compute_distance_stable(anc_ptr, pos_ptr, D, p);
        double d_n = compute_distance_stable(anc_ptr, neg_ptr, D, p);

        if (swap) {
            double d_s = compute_distance_stable(pos_ptr, neg_ptr, D, p);
            if (d_s < d_n) d_n = d_s;
        }

        double loss = std::max(0.0, d_p - d_n + margin);
        out_ptr[0] = utils::cast<T>(static_cast<float>(loss));
        return;
    }

    // N > 1 的确定性处理
    std::vector<double> losses(N);

    #pragma omp parallel for schedule(static)
    for (size_t n = 0; n < N; ++n) {
        const T* a_row = anc_ptr + n * D;
        const T* p_row = pos_ptr + n * D;
        const T* n_row = neg_ptr + n * D;

        double d_p = compute_distance_stable(a_row, p_row, D, p);
        double d_n = compute_distance_stable(a_row, n_row, D, p);

        if (swap) {
            double d_s = compute_distance_stable(p_row, n_row, D, p);
            if (d_s < d_n) d_n = d_s;
        }

        losses[n] = std::max(0.0, d_p - d_n + margin);
    }

    if (reduction == 0) { // None
        for (size_t n = 0; n < N; ++n) {
            out_ptr[n] = utils::cast<T>(static_cast<float>(losses[n]));
        }
    } else { 
        double total_loss = 0.0;
        // 顺序求和
        for (double l : losses) {
            total_loss += l;
        }

        if (reduction == 1) { // Mean
            total_loss /= static_cast<double>(N);
        }
        out_ptr[0] = utils::cast<T>(static_cast<float>(total_loss));
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *anchor,
    const void *positive,
    const void *negative,
    void *stream) const {

    auto dtype = _info.dtype();

    switch (dtype) {
    case INFINI_DTYPE_F32:
        cpu::calculate_cpu_impl<float>(_info, output, anchor, positive, negative);
        break;
    case INFINI_DTYPE_F64:
        cpu::calculate_cpu_impl<double>(_info, output, anchor, positive, negative);
        break;
    case INFINI_DTYPE_F16:
        cpu::calculate_cpu_impl<fp16_t>(_info, output, anchor, positive, negative);
        break;
    case INFINI_DTYPE_BF16:
        cpu::calculate_cpu_impl<bf16_t>(_info, output, anchor, positive, negative);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::triplet_margin_loss::cpu