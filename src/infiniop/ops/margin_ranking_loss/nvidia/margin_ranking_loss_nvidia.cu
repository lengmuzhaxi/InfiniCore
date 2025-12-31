#include "margin_ranking_loss_nvidia.cuh"
#include "../cuda/kernel.cuh"
#include "../../../handle.h"
// 引入 NVIDIA Handle 定义
#include "../../../devices/nvidia/nvidia_handle.h" 

#include <cstdint>
#include <vector>
#include <algorithm>
#include <numeric>

namespace op::margin_ranking_loss::nvidia {

// ==================================================================
// 1. 定义公共结构体 (解决 Opaque private 访问权限问题)
// ==================================================================
struct MarginRankingLossOpaqueData {
    int ndim;
    size_t numel; // 广播后的元素总数

    // Device Pointers (GPU 显存指针)
    int64_t* d_shape = nullptr;      // Output Shape
    int64_t* d_strides1 = nullptr;   // Input1 Strides
    int64_t* d_strides2 = nullptr;   // Input2 Strides
    int64_t* d_strides_tar = nullptr;// Target Strides
};

// 让 Descriptor::Opaque 继承自公共数据结构
struct Descriptor::Opaque : public MarginRankingLossOpaqueData {};

Descriptor::~Descriptor() {
    if (_opaque) {
        if (_opaque->d_shape) cudaFree(_opaque->d_shape);
        if (_opaque->d_strides1) cudaFree(_opaque->d_strides1);
        if (_opaque->d_strides2) cudaFree(_opaque->d_strides2);
        if (_opaque->d_strides_tar) cudaFree(_opaque->d_strides_tar);
        delete _opaque;
        _opaque = nullptr;
    }
}

// ==================================================================
// 2. 辅助函数: 步长计算与显存上传
// ==================================================================

// 计算广播步长 (Host 端)
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
            // 广播规则: 如果输入维度为1，则步长设为0；否则使用原步长
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

// 上传数据到 GPU
template <typename T>
static T* upload_to_device(const std::vector<T>& host_vec) {
    if (host_vec.empty()) return nullptr;
    T* d_ptr = nullptr;
    size_t size_bytes = host_vec.size() * sizeof(T);
    cudaMalloc(&d_ptr, size_bytes);
    cudaMemcpy(d_ptr, host_vec.data(), size_bytes, cudaMemcpyHostToDevice);
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

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    auto out_ptr = reinterpret_cast<T *>(output);
    auto in1_ptr = reinterpret_cast<const T *>(input1);
    auto in2_ptr = reinterpret_cast<const T *>(input2);
    auto tar_ptr = reinterpret_cast<const T *>(target);

    // 构造 Functor
    op::margin_ranking_loss::cuda::MarginRankingLossFunctor functor(info.margin());
    int reduction = info.reduction();
    size_t numel = opaque->numel;
    int ndim = opaque->ndim;

    if (numel == 0) return;

    size_t block_size = 256;
    size_t grid_size = std::min((numel + block_size - 1) / block_size, static_cast<size_t>(2048));

    // ------------------------------------------
    // 模式 1: Elementwise (Reduction = None)
    // ------------------------------------------
    if (reduction == 0) {
        op::margin_ranking_loss::cuda::margin_ranking_loss_kernel<T>
            <<<grid_size, block_size, 0, cuda_stream>>>(
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
    // 模式 2: Reduction (Mean / Sum)
    // ------------------------------------------
    else {
        // 使用 workspace 作为 float 累加器
        float* acc_ptr = reinterpret_cast<float*>(workspace);
        cudaMemsetAsync(acc_ptr, 0, sizeof(float), cuda_stream);
        
        float scale = (reduction == 1) ? (1.0f / static_cast<float>(numel)) : 1.0f; // 1=Mean, 2=Sum

        op::margin_ranking_loss::cuda::margin_ranking_loss_reduce_kernel<T>
            <<<grid_size, block_size, 0, cuda_stream>>>(
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
        
        // 将结果转回 T 类型并写入 output
        op::margin_ranking_loss::cuda::cast_float_to_t<T>
            <<<1, 1, 0, cuda_stream>>>(out_ptr, acc_ptr);
    }
}

// ==================================================================
// 4. Descriptor::create 实现
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

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);

    // 1. 创建 Info
    auto result = MarginRankingLossInfo::create(out_desc, input1_desc, input2_desc, target_desc, margin, p, reduction);
    CHECK_RESULT(result);
    auto info = result.take();

    // 2. 准备 Opaque (计算广播形状与步长)
    auto opaque = new Opaque();

    // 手动推导最大形状 (因为 reduction 模式下 out_desc 是标量，不能用于遍历)
    int ndim = std::max({(int)input1_desc->ndim(), (int)input2_desc->ndim(), (int)target_desc->ndim()});
    std::vector<size_t> broadcast_shape(ndim, 1);

    for (int i = 0; i < ndim; ++i) {
        // 从右向左对齐获取各维度大小
        size_t d1 = (i < ndim - (int)input1_desc->ndim()) ? 1 : input1_desc->shape()[i - (ndim - input1_desc->ndim())];
        size_t d2 = (i < ndim - (int)input2_desc->ndim()) ? 1 : input2_desc->shape()[i - (ndim - input2_desc->ndim())];
        size_t d3 = (i < ndim - (int)target_desc->ndim()) ? 1 : target_desc->shape()[i - (ndim - target_desc->ndim())];
        broadcast_shape[i] = std::max({d1, d2, d3});
    }

    // 设置基础信息
    opaque->ndim = ndim;
    size_t numel = 1;
    for (auto s : broadcast_shape) numel *= s;
    opaque->numel = numel;

    // 转换 Shape 为 int64 vector 并上传
    std::vector<int64_t> host_shape(broadcast_shape.begin(), broadcast_shape.end());
    opaque->d_shape = upload_to_device(host_shape);

    // 计算并上传 Strides
    auto strides1 = compute_broadcast_strides(broadcast_shape, input1_desc);
    opaque->d_strides1 = upload_to_device(strides1);

    auto strides2 = compute_broadcast_strides(broadcast_shape, input2_desc);
    opaque->d_strides2 = upload_to_device(strides2);

    auto strides_tar = compute_broadcast_strides(broadcast_shape, target_desc);
    opaque->d_strides_tar = upload_to_device(strides_tar);

    // 3. 确定 Workspace 大小
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
// 5. Descriptor::calculate 实现
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

    // 检查 workspace
    if (reduction != 0 && workspace_size < sizeof(float)) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    switch (dtype) {
    case INFINI_DTYPE_F16:
        launch_kernel<half>(output, input1, input2, target, workspace, _info, _opaque, stream);
        break;
    case INFINI_DTYPE_BF16:
        launch_kernel<nv_bfloat16>(output, input1, input2, target, workspace, _info, _opaque, stream);
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

} // namespace op::margin_ranking_loss::nvidia