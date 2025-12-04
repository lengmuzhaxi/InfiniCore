#include "addbmm_nvidia.cuh" // 包含 Kernel 定义
#include "../cuda/kernel.cuh" // 包含 Descriptor 定义
#include "../../../handle.h" 
#include <vector>

namespace op::addbmm::nvidia {

// ==================================================================
// Kernel Launch Helper
// ==================================================================
template <typename T>
void launch_kernel(
    void *output, 
    const void *input, 
    const void *batch1, 
    const void *batch2, 
    size_t b,
    size_t n,
    size_t m,
    size_t p,
    float alpha,
    float beta,
    const AddbmmInfo &info, // 【关键修改】传入 Info 对象以获取 strides
    void *stream) {
    
    auto out_ptr = reinterpret_cast<T *>(output);
    auto in_ptr = reinterpret_cast<const T *>(input);
    auto b1_ptr = reinterpret_cast<const T *>(batch1);
    auto b2_ptr = reinterpret_cast<const T *>(batch2);
    
    // 【关键修改】从 Info 中提取 Strides
    const auto& os = info.out_strides();
    const auto& is = info.in_strides();
    const auto& b1s = info.b1_strides();
    const auto& b2s = info.b2_strides();
    
    // 计算总输出元素个数: n * p
    size_t total_elements = n * p;
    
    // CUDA Grid/Block 配置
    size_t block_size = 256;
    size_t grid_size = (total_elements + block_size - 1) / block_size;

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // 调用 addbmm_nvidia.cuh 中定义的 kernel
    // 【关键修改】传递 strides 参数
    addbmm_kernel<T><<<grid_size, block_size, 0, cuda_stream>>>(
        out_ptr,
        in_ptr,
        b1_ptr,
        b2_ptr,
        b, n, m, p,
        alpha, beta,
        // 传递 Strides:
        os[0], os[1],           // Output [n, p]
        is[0], is[1],           // Input [n, p]
        b1s[0], b1s[1], b1s[2], // Batch1 [b, n, m]
        b2s[0], b2s[1], b2s[2]  // Batch2 [b, m, p]
    );
}

// ==================================================================
// Descriptor Implementation
// ==================================================================

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec,
    float alpha,
    float beta) {

    // 0. 检查输入数量
    if (input_desc_vec.size() != 3) {
        return INFINI_STATUS_BAD_PARAM;
    }

    // 1. 使用 Info 类解析参数 (Info::create 现在会解析 strides)
    auto info_result = AddbmmInfo::create(
        out_desc, 
        input_desc_vec[0], // input
        input_desc_vec[1], // batch1
        input_desc_vec[2], // batch2
        alpha,
        beta
    );

    if (!info_result) {
        return info_result.status();
    }
    
    // 2. 创建 Descriptor
    *desc_ptr = new Descriptor(
        new Opaque(),        // Opaque 指针
        info_result.take(),  // Info 对象
        0,                   // Workspace size
        handle->device,      // Device Type
        handle->device_id    // Device ID
    );

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    // 检查输入指针数量
    if (inputs.size() != 3) {
        return INFINI_STATUS_BAD_PARAM;
    }

    const void *input_ptr = inputs[0];
    const void *batch1_ptr = inputs[1];
    const void *batch2_ptr = inputs[2];

    // 从 Info 对象中提取参数
    auto dtype = _info.dtype();
    size_t b = _info.b();
    size_t n = _info.n();
    size_t m = _info.m();
    size_t p = _info.p();
    float alpha = _info.alpha();
    float beta = _info.beta();

    switch (dtype) {
    case INFINI_DTYPE_F16:
        // 【关键修改】传入 _info
        launch_kernel<half>(output, input_ptr, batch1_ptr, batch2_ptr, 
                            b, n, m, p, alpha, beta, _info, stream);
        break;
    case INFINI_DTYPE_BF16:
        // 使用标准类型 nv_bfloat16
        launch_kernel<nv_bfloat16>(output, input_ptr, batch1_ptr, batch2_ptr, 
                                   b, n, m, p, alpha, beta, _info, stream);
        break;
    case INFINI_DTYPE_F32:
        launch_kernel<float>(output, input_ptr, batch1_ptr, batch2_ptr, 
                             b, n, m, p, alpha, beta, _info, stream);
        break;
    case INFINI_DTYPE_F64:
        launch_kernel<double>(output, input_ptr, batch1_ptr, batch2_ptr, 
                              b, n, m, p, alpha, beta, _info, stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::addbmm::nvidia