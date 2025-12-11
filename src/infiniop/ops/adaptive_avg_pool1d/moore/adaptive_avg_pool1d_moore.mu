#include "adaptive_avg_pool1d_moore.h" // 必须包含 API 头文件以获取 Descriptor 定义
#include "adaptive_avg_pool1d_moore_kernel.h" // 如果有单独的 kernel 头文件

#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>
#include <type_traits> // for std::is_same_v

#include "../../../devices/moore/moore_handle.h"

namespace op::adaptive_avg_pool1d::moore {

// ==================================================================
// 1. Kernel Implementation
// ==================================================================

template <typename T>
__global__ void adaptive_avg_pool1d_kernel(
    const int total_elements,
    const int input_size,
    const int output_size,
    const T *input,
    T *output) {
    
    // idx 对应输出张量扁平化后的索引 (N * C * L_out)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        // 1. 解析索引
        // 输出形式为 (N, C, L_out)，可以看作 (Batch*Channel, L_out)
        // w_out: 当前在输出序列长度 (L_out) 中的位置
        int w_out = idx % output_size;
        // bc: 当前属于哪个 Batch 和 Channel (Batch * Channel)
        int bc = idx / output_size;

        // 2. 计算输入对应的窗口范围 [start, end)
        // 公式: start = floor(w_out * L_in / L_out)
        //       end   = ceil((w_out + 1) * L_in / L_out)
        int start = (w_out * input_size) / output_size;
        int end = ((w_out + 1) * input_size + output_size - 1) / output_size;

        // 边界防御
        start = (start < 0) ? 0 : start;
        end = (end > input_size) ? input_size : end;

        int kernel_size = end - start;
        if (kernel_size < 1) kernel_size = 1;

        // 3. 定位输入指针
        // 输入形式 (N, C, L_in)，即 (Batch*Channel, L_in)
        // 当前通道的起始地址偏移量 = bc * input_size
        const T *in_ptr = input + bc * input_size;

        // 4. 累加计算 (强制使用 float)
        float sum = 0.0f;

        for (int i = start; i < end; ++i) {
            T val = in_ptr[i];
            
            if constexpr (std::is_same_v<T, half>) {
                sum += __half2float(val);
            } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
                sum += __bfloat162float(val);
            } else {
                sum += static_cast<float>(val);
            }
        }

        // 5. 计算平均值并写回
        float avg = sum / static_cast<float>(kernel_size);

        if constexpr (std::is_same_v<T, half>) {
            output[idx] = __float2half(avg);
        } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
            output[idx] = __float2bfloat16(avg);
        } else {
            output[idx] = static_cast<T>(avg);
        }
    }
}

// ==================================================================
// 2. Launcher Implementation
// ==================================================================

template <typename T>
void adaptive_avg_pool1d_moore_launch(
    const AdaptiveAvgPool1dInfo &info,
    T *output,
    const T *input,
    void *stream) {
    
    // Info 类通常会将 Batch 和 Channel 合并视为 num_channels (如果维度为3)
    // input_size = L_in, output_size = L_out
    int input_size = info.input_size();
    int output_size = info.output_size();
    
    // 总输出元素数量 = N * C * L_out
    // info.num_channels() 通常包含 Batch * C
    size_t total_elements = info.num_channels() * output_size;

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    adaptive_avg_pool1d_kernel<T><<<blocks, threads, 0, (musaStream_t)stream>>>(
        total_elements,
        input_size,
        output_size,
        input,
        output
    );
}

// ==================================================================
// 3. Descriptor Implementation
// ==================================================================

Descriptor::~Descriptor() = default;

// 注意：如果您的 API 头文件中使用的是通用宏 (类似 addbmm 那个情况)，
// 且宏定义了 std::vector<...> input_desc_vec，则需要修改此函数签名。
// 鉴于 AdaptivePool 通常是单输入单输出，这里采用标准签名。
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);

    auto info_result = AdaptiveAvgPool1dInfo::create(out_desc, in_desc);
    
    if (!info_result) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    *desc_ptr = new Descriptor(
        nullptr,
        *info_result,
        0, // No workspace needed
        handle->device,
        handle->device_id
    );

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    switch (_info.dtype()) {
    case INFINI_DTYPE_F16:
        adaptive_avg_pool1d_moore_launch<half>(
            _info, 
            static_cast<half *>(output), 
            static_cast<const half *>(input), 
            stream);
        break;
        
    case INFINI_DTYPE_BF16:
        adaptive_avg_pool1d_moore_launch<__mt_bfloat16>(
            _info, 
            static_cast<__mt_bfloat16 *>(output), 
            static_cast<const __mt_bfloat16 *>(input), 
            stream);
        break;

    case INFINI_DTYPE_F32:
        adaptive_avg_pool1d_moore_launch<float>(
            _info, 
            static_cast<float *>(output), 
            static_cast<const float *>(input), 
            stream);
        break;

    case INFINI_DTYPE_F64:
        adaptive_avg_pool1d_moore_launch<double>(
            _info, 
            static_cast<double *>(output), 
            static_cast<const double *>(input), 
            stream);
        break;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::adaptive_avg_pool1d::moore