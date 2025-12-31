#include "margin_ranking_loss_moore.h"
#include "margin_ranking_loss_moore_kernel.h"
#include "../../../handle.h"
#include "../../../devices/moore/moore_handle.h"

#include <cstdint>
#include <vector>
#include <algorithm>

namespace op::margin_ranking_loss::moore {

// ==================================================================
// 1. Define Public Structure
// ==================================================================
struct MarginRankingLossOpaqueData {
    int ndim;
    size_t numel; // Total elements after broadcasting

    // Device Pointers
    int64_t* d_shape = nullptr;      // Output Shape
    int64_t* d_strides1 = nullptr;   // Input1 Strides
    int64_t* d_strides2 = nullptr;   // Input2 Strides
    int64_t* d_strides_tar = nullptr;// Target Strides
};

struct Descriptor::Opaque : public MarginRankingLossOpaqueData {};

Descriptor::~Descriptor() {
    if (_opaque) {
        if (_opaque->d_shape) musaFree(_opaque->d_shape);
        if (_opaque->d_strides1) musaFree(_opaque->d_strides1);
        if (_opaque->d_strides2) musaFree(_opaque->d_strides2);
        if (_opaque->d_strides_tar) musaFree(_opaque->d_strides_tar);
        delete _opaque;
        _opaque = nullptr;
    }
}

// ==================================================================
// 2. Helper Functions: Stride Calculation & Upload
// ==================================================================

static std::vector<int64_t> compute_broadcast_strides(
    const std::vector<size_t>& out_shape,
    infiniopTensorDescriptor_t input_desc) {
    
    int out_ndim = static_cast<int>(out_shape.size());
    
    const auto& in_shape = input_desc->shape();
    const auto& in_strides = input_desc->strides();
    
    std::vector<int64_t> effective_strides(out_ndim, 0);

    for (int i = 0; i < out_ndim; ++i) {
        int out_idx = out_ndim - 1 - i;
        int in_idx = (int)input_desc->ndim() - 1 - i;

        if (in_idx >= 0) {
            size_t dim_size = in_shape[in_idx];
            if (dim_size == 1) {
                effective_strides[out_idx] = 0;
            } else {
                effective_strides[out_idx] = in_strides[in_idx];
            }
        } else {
            effective_strides[out_idx] = 0;
        }
    }
    return effective_strides;
}

template <typename T>
static T* upload_to_device(const std::vector<T>& host_vec) {
    if (host_vec.empty()) return nullptr;
    T* d_ptr = nullptr;
    size_t size_bytes = host_vec.size() * sizeof(T);
    musaMalloc(&d_ptr, size_bytes);
    musaMemcpy(d_ptr, host_vec.data(), size_bytes, musaMemcpyHostToDevice);
    return d_ptr;
}

// ==================================================================
// 3. Kernel Launch Logic
// ==================================================================

template <typename T>
void launch_kernel(
    void *output,
    const void *input1,
    const void *input2,
    const void *target,
    void* workspace,
    const MarginRankingLossInfo& info,
    const MarginRankingLossOpaqueData* opaque,
    void *stream) {

    auto musa_stream = reinterpret_cast<musaStream_t>(stream);

    auto out_ptr = reinterpret_cast<T *>(output);
    auto in1_ptr = reinterpret_cast<const T *>(input1);
    auto in2_ptr = reinterpret_cast<const T *>(input2);
    auto tar_ptr = reinterpret_cast<const T *>(target);

    MarginRankingLossFunctor functor(info.margin());

    int reduction = info.reduction();
    size_t numel = opaque->numel;
    int ndim = opaque->ndim;

    if (numel == 0) return;

    size_t block_size = 256;
    size_t grid_size = std::min((numel + block_size - 1) / block_size, static_cast<size_t>(2048));

    // ------------------------------------------
    // Mode 1: Elementwise (Reduction = None)
    // ------------------------------------------
    if (reduction == 0) {
        margin_ranking_loss_kernel<T>
            <<<grid_size, block_size, 0, musa_stream>>>(
                out_ptr,
                in1_ptr,
                in2_ptr,
                tar_ptr,
                numel,
                ndim,
                opaque->d_shape,
                opaque->d_strides1,
                opaque->d_strides2,
                opaque->d_strides_tar,
                functor
            );
    } 
    // ------------------------------------------
    // Mode 2: Reduction (Mean / Sum)
    // ------------------------------------------
    else {
        float* acc_ptr = reinterpret_cast<float*>(workspace);
        musaMemsetAsync(acc_ptr, 0, sizeof(float), musa_stream);
        
        float scale = (reduction == 1) ? (1.0f / static_cast<float>(numel)) : 1.0f; // 1=Mean, 2=Sum

        margin_ranking_loss_reduce_kernel<T>
            <<<grid_size, block_size, 0, musa_stream>>>(
                acc_ptr,
                in1_ptr,
                in2_ptr,
                tar_ptr,
                numel,
                ndim,
                opaque->d_shape,
                opaque->d_strides1,
                opaque->d_strides2,
                opaque->d_strides_tar,
                functor,
                scale
            );
        
        cast_float_to_t<T>
            <<<1, 1, 0, musa_stream>>>(out_ptr, acc_ptr);
    }
}

// ==================================================================
// 4. Descriptor::create Implementation
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

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);

    auto result = MarginRankingLossInfo::create(out_desc, input1_desc, input2_desc, target_desc, margin, p, reduction);
    if (!result) return result.status();
    auto info = result.take();

    auto opaque = new Opaque();

    int ndim = std::max({(int)input1_desc->ndim(), (int)input2_desc->ndim(), (int)target_desc->ndim()});
    std::vector<size_t> broadcast_shape(ndim, 1);

    for (int i = 0; i < ndim; ++i) {
        size_t d1 = (i < ndim - (int)input1_desc->ndim()) ? 1 : input1_desc->shape()[i - (ndim - input1_desc->ndim())];
        size_t d2 = (i < ndim - (int)input2_desc->ndim()) ? 1 : input2_desc->shape()[i - (ndim - input2_desc->ndim())];
        size_t d3 = (i < ndim - (int)target_desc->ndim()) ? 1 : target_desc->shape()[i - (ndim - target_desc->ndim())];
        broadcast_shape[i] = std::max({d1, d2, d3});
    }

    opaque->ndim = ndim;
    size_t numel = 1;
    for (auto s : broadcast_shape) numel *= s;
    opaque->numel = numel;

    std::vector<int64_t> host_shape(broadcast_shape.begin(), broadcast_shape.end());
    opaque->d_shape = upload_to_device(host_shape);

    auto strides1 = compute_broadcast_strides(broadcast_shape, input1_desc);
    opaque->d_strides1 = upload_to_device(strides1);

    auto strides2 = compute_broadcast_strides(broadcast_shape, input2_desc);
    opaque->d_strides2 = upload_to_device(strides2);

    auto strides_tar = compute_broadcast_strides(broadcast_shape, target_desc);
    opaque->d_strides_tar = upload_to_device(strides_tar);

    size_t workspace_size = 0;
    if (reduction != 0) {
        workspace_size = sizeof(float);
    }

    *desc_ptr = new Descriptor(
        opaque,
        info,
        workspace_size,
        handle->device,
        handle->device_id
    );

    return INFINI_STATUS_SUCCESS;
}

// ==================================================================
// 5. Descriptor::calculate Implementation
// ==================================================================
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input1,
    const void *input2,
    const void *target,
    void *stream) const {

    auto dtype = _info.dtype();
    int reduction = _info.reduction();

    if (reduction != 0 && workspace_size < sizeof(float)) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    switch (dtype) {
    case INFINI_DTYPE_F16:
        // 【关键修正】FP16 使用 __half
        launch_kernel<__half>(output, input1, input2, target, workspace, _info, _opaque, stream);
        break;
    case INFINI_DTYPE_BF16:
        // 【关键修正】BF16 使用 __mt_bfloat16
        launch_kernel<__mt_bfloat16>(output, input1, input2, target, workspace, _info, _opaque, stream);
        break;
    case INFINI_DTYPE_F32:
        launch_kernel<float>(output, input1, input2, target, workspace, _info, _opaque, stream);
        break;
    case INFINI_DTYPE_F64:
        launch_kernel<double>(output, input1, input2, target, workspace, _info, _opaque, stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::margin_ranking_loss::moore