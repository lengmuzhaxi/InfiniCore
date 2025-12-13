#include "matrix_power_cpu.h"
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include "../../../devices/cpu/common_cpu.h"
#include "../../../handle.h"

// 根据你的目录结构 src/infiniop/ops/matrix_power/cpu/
// 需要向上跳 4 层才能到达 src/utils
#include "../../../../utils/custom_types.h"

namespace op::matrix_power::cpu {

Descriptor::~Descriptor() = default;

// ==================================================================
// 辅助工具：基于 utils::cast 的安全读写
// ==================================================================

// 写入数值：dst[offset] = val
// 使用 utils::cast 处理 float -> fp16/bf16/float/double 的转换
template <typename T>
inline void set_val(T* data, size_t offset, float val) {
    // utils::cast 会自动处理 float 到 CustomFloat16/CustomBFloat16 的转换逻辑
    data[offset] = utils::cast<T>(val);
}

// 读取数值：return src[offset]
// 使用 utils::cast 处理 fp16/bf16/float/double -> float 的转换
template <typename T>
inline float get_val(const T* data, size_t offset) {
    return utils::cast<float>(data[offset]);
}

// 复制数值：dst[dst_off] = src[src_off]
// CustomFloat16/BFloat16 是 POD 结构体 (只包含 uint16_t _v)，直接赋值是安全的
template <typename T>
inline void copy_val(T* dst, size_t dst_off, const T* src, size_t src_off) {
    dst[dst_off] = src[src_off];
}

// ==================================================================
// 矩阵操作核心逻辑
// ==================================================================

// 定义计算精度：Double 用 double 计算，其他(fp16/bf16/float)都用 float 计算
template <typename T> struct ComputeType { using type = float; };
template <> struct ComputeType<double> { using type = double; };

// 设置为单位矩阵
template <typename T>
void set_identity(T *data, size_t m, int64_t stride_row, int64_t stride_col) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            float val = (i == j) ? 1.0f : 0.0f;
            set_val(data, i * stride_row + j * stride_col, val);
        }
    }
}

// 矩阵复制
template <typename T>
void copy_matrix(T *dst, const T *src, size_t m, 
                 int64_t dst_stride_row, int64_t dst_stride_col,
                 int64_t src_stride_row, int64_t src_stride_col) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            copy_val(dst, i * dst_stride_row + j * dst_stride_col,
                     src, i * src_stride_row + j * src_stride_col);
        }
    }
}

// 矩阵乘法: Dst = SrcA * SrcB
// 使用 ComputeT (float/double) 类型的 workspace 作为中间累加器
template <typename T, typename ComputeT>
void matmul(T *dst, const T *src_a, const T *src_b, ComputeT *workspace, size_t m,
            int64_t dst_stride_row, int64_t dst_stride_col,
            int64_t a_stride_row, int64_t a_stride_col,
            int64_t b_stride_row, int64_t b_stride_col) {
    
    // 1. 初始化 workspace (必须清零，因为后续是累加)
    // workspace 是连续的 m*m 内存
    for (size_t i = 0; i < m * m; ++i) {
        workspace[i] = static_cast<ComputeT>(0);
    }

    // 2. 矩阵乘法 (Accumulate)
    for (size_t i = 0; i < m; ++i) {
        for (size_t k = 0; k < m; ++k) {
            // 从 A 读取并转为高精度
            ComputeT a = static_cast<ComputeT>(get_val(src_a, i * a_stride_row + k * a_stride_col));
            for (size_t j = 0; j < m; ++j) {
                // 从 B 读取并转为高精度
                ComputeT b = static_cast<ComputeT>(get_val(src_b, k * b_stride_row + j * b_stride_col));
                workspace[i * m + j] += a * b;
            }
        }
    }

    // 3. 将结果写回目标矩阵 (Scatter)
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            // 转回目标类型 T 并写入
            set_val(dst, i * dst_stride_row + j * dst_stride_col, static_cast<float>(workspace[i * m + j]));
        }
    }
}

// 核心计算入口
template <typename Tdata>
void calculate_impl(
    const MatrixPowerInfo &info,
    void *workspace,
    void *output,
    const void *input,
    int n_exponent) {

    // 确定中间计算类型 (FP16/BF16 -> Float)
    using CompT = typename ComputeType<Tdata>::type;

    size_t batch_total = info.batch_size();
    size_t m = info.m();
    size_t ndim = info.ndim();
    const auto &shape = info.shape();
    const auto &in_strides = info.in_strides();
    const auto &out_strides = info.out_strides();

    // 提取最后两维的 stride
    int64_t in_s_row = in_strides[ndim - 2];
    int64_t in_s_col = in_strides[ndim - 1];
    int64_t out_s_row = out_strides[ndim - 2];
    int64_t out_s_col = out_strides[ndim - 1];

    Tdata *out_ptr_base = reinterpret_cast<Tdata *>(output);
    const Tdata *inp_ptr_base = reinterpret_cast<const Tdata *>(input);
    CompT *work_buf = reinterpret_cast<CompT *>(workspace); 

    for (size_t b = 0; b < batch_total; ++b) {
        size_t linear_idx = b;
        size_t in_offset = 0;
        size_t out_offset = 0;
        
        // 计算 Batch 维度的偏移量
        if (ndim > 2) {
            size_t temp_idx = linear_idx;
            for (int d = static_cast<int>(ndim) - 3; d >= 0; --d) {
                size_t dim_idx = temp_idx % shape[d];
                temp_idx /= shape[d];
                in_offset += dim_idx * in_strides[d];
                out_offset += dim_idx * out_strides[d];
            }
        }

        Tdata *curr_out = out_ptr_base + out_offset;
        const Tdata *curr_in = inp_ptr_base + in_offset;

        if (n_exponent == 0) {
            // Case 0: I
            set_identity(curr_out, m, out_s_row, out_s_col);
        } else if (n_exponent == 1) {
            // Case 1: Copy
            copy_matrix(curr_out, curr_in, m, out_s_row, out_s_col, in_s_row, in_s_col);
        } else {
            // Case N>1: 线性迭代
            // 1. 初始化 Result = Input
            copy_matrix(curr_out, curr_in, m, out_s_row, out_s_col, in_s_row, in_s_col);
            
            // 2. 循环: Result = Result * Input
            // 每次循环使用 workspace 存储中间结果，避免原地修改导致的数值污染
            for (int i = 0; i < n_exponent - 1; ++i) {
                matmul<Tdata, CompT>(
                    curr_out,   // Dst (Result)
                    curr_out,   // Src A (Result)
                    curr_in,    // Src B (Input)
                    work_buf,   // Workspace
                    m,
                    out_s_row, out_s_col,
                    out_s_row, out_s_col,
                    in_s_row, in_s_col
                );
            }
        }
    }
}

// ==================================================================
// Descriptor 接口实现
// ==================================================================

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc) {
    
    MatrixPowerInfo info;
    auto status = MatrixPowerInfo::create(out_desc, in_desc, &info);
    if (status != INFINI_STATUS_SUCCESS) {
        return status;
    }

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    
    // 计算 Workspace 大小: m * m * sizeof(ComputeType)
    // 即使是 FP16 输入，我们也用 Float (4字节) 做中间计算
    size_t m = info.m();
    size_t dt_compute_size = (info.dtype() == INFINI_DTYPE_F64) ? sizeof(double) : sizeof(float);
    size_t workspace_size = m * m * dt_compute_size;

    *desc_ptr = new Descriptor(
        nullptr, std::move(info), workspace_size, 
        handle->device, handle->device_id
    );

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    int n,
    void *stream) const {

    if (n < 0) return INFINI_STATUS_BAD_PARAM;
    
    // 安全检查：Workspace 大小
    if (workspace_size < _workspace_size) return INFINI_STATUS_BAD_PARAM;
    // 安全检查：指针非空 (如果需要 workspace)
    if (_workspace_size > 0 && workspace == nullptr) return INFINI_STATUS_BAD_PARAM;
    if (output == nullptr || input == nullptr) return INFINI_STATUS_BAD_PARAM;

    switch (_info.dtype()) {
    case INFINI_DTYPE_F16:
        calculate_impl<fp16_t>(_info, workspace, output, input, n);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_BF16:
        calculate_impl<bf16_t>(_info, workspace, output, input, n);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F32:
        calculate_impl<float>(_info, workspace, output, input, n);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F64:
        calculate_impl<double>(_info, workspace, output, input, n);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::matrix_power::cpu