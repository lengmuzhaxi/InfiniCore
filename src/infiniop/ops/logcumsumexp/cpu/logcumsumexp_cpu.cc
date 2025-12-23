#include "logcumsumexp_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>
#include <numeric>

#include "../../../../utils/custom_types.h"

namespace op::logcumsumexp::cpu {
struct Descriptor::Opaque {
    int ndim;
    std::vector<int64_t> shape;
    std::vector<int64_t> x_strides;
    std::vector<int64_t> y_strides;
    int axis; // 记录操作的原始维度
};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
        _opaque = nullptr;
    }
}

static void extract_meta(infiniopTensorDescriptor_t desc, 
                         std::vector<int64_t>& shape, 
                         std::vector<int64_t>& strides, 
                         int& ndim) {
    ndim = desc->ndim;
    // 假设 desc->dims 和 desc->strides 是数组指针
    // 如果是 int* 或 int64_t*，vector 的 assign 可以直接拷贝
    shape.assign(desc->dims, desc->dims + ndim);
    strides.assign(desc->strides, desc->strides + ndim);
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    int axis,
    int exclusive,
    int reverse) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    
    // 创建基础 Info 对象
    auto result = LogCumSumExpInfo::create(y_desc, x_desc, axis, exclusive, reverse);
    CHECK_RESULT(result);

    // -----------------------------------------------------------
    // 初始化 Opaque 数据，保存 Shape 和 Stride
    // -----------------------------------------------------------
    auto opaque = new Opaque();
    opaque->axis = axis; // 保存原始 axis

    // 提取输入 X 的信息
    extract_meta(x_desc, opaque->shape, opaque->x_strides, opaque->ndim);
    
    // 提取输出 Y 的 stride (Y 的 shape 和 X 一样)
    int dummy_ndim;
    std::vector<int64_t> dummy_shape;
    extract_meta(y_desc, dummy_shape, opaque->y_strides, dummy_ndim);

    *desc_ptr = new Descriptor(
        opaque,
        result.take(),
        0, 
        handle->device, 
        handle->device_id
    );

    return INFINI_STATUS_SUCCESS;
}
inline size_t get_offset_from_flat_index(size_t flat_index, 
                                         int dim_start, int dim_end, 
                                         const std::vector<int64_t>& shape, 
                                         const std::vector<int64_t>& strides) {
    size_t offset = 0;
    size_t temp_idx = flat_index;

    // 从这一段维度的"最内层"开始反解坐标
    // 例如：dims=[2,3], strides=[30,10]. flat_index=4 (即坐标 1,1)
    // d=1: coord = 4 % 3 = 1; temp = 4 / 3 = 1; offset += 1 * 10
    // d=0: coord = 1 % 2 = 1; temp = 1 / 2 = 0; offset += 1 * 30
    for (int d = dim_end - 1; d >= dim_start; --d) {
        size_t size = shape[d];
        size_t coord = temp_idx % size;
        temp_idx /= size;
        offset += coord * strides[d];
    }
    return offset;
}

template <typename T>
void calculate_cpu_impl(
    const LogCumSumExpInfo &info,
    const Descriptor::Opaque *meta, // 传入 Opaque
    void *y,
    const void *x) {

    size_t outer_size = info.outer_size();
    size_t axis_size = info.axis_size();
    size_t inner_size = info.inner_size();
    bool exclusive = info.exclusive();
    bool reverse = info.reverse();
    
    int axis = meta->axis;
    int ndim = meta->ndim;

    auto y_ptr = reinterpret_cast<T *>(y);
    auto x_ptr = reinterpret_cast<const T *>(x);

    // 获取 axis 维度的 stride，用于最内层循环的快速移动
    int64_t x_axis_stride = meta->x_strides[axis];
    int64_t y_axis_stride = meta->y_strides[axis];

    // OMP 并行：并行处理所有独立的行/列
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < outer_size; ++i) {
        for (size_t j = 0; j < inner_size; ++j) {
            
            // -------------------------------------------------------
            // 关键修复：计算 Base Offset
            // -------------------------------------------------------
            // i 对应维度 [0, axis) 的组合
            size_t x_outer = get_offset_from_flat_index(i, 0, axis, meta->shape, meta->x_strides);
            size_t y_outer = get_offset_from_flat_index(i, 0, axis, meta->shape, meta->y_strides);

            // j 对应维度 [axis+1, ndim) 的组合
            size_t x_inner = get_offset_from_flat_index(j, axis + 1, ndim, meta->shape, meta->x_strides);
            size_t y_inner = get_offset_from_flat_index(j, axis + 1, ndim, meta->shape, meta->y_strides);

            size_t x_base = x_outer + x_inner;
            size_t y_base = y_outer + y_inner;

            // -------------------------------------------------------
            // LogCumSumExp 核心逻辑 (数值稳定版)
            // -------------------------------------------------------
            double running_max = -std::numeric_limits<double>::infinity();
            double running_sum_exp = 0.0;

            for (size_t k = 0; k < axis_size; ++k) {
                // 处理 reverse
                size_t k_idx = reverse ? (axis_size - 1 - k) : k;
                
                // 使用 stride 计算当前元素的真实地址
                size_t x_curr = x_base + k_idx * x_axis_stride;
                size_t y_curr = y_base + k_idx * y_axis_stride;

                float val = utils::cast<float>(x_ptr[x_curr]);
                
                // Case: Exclusive (Scan 不包含当前元素)
                if (exclusive) {
                    if (running_sum_exp == 0.0) {
                        y_ptr[y_curr] = utils::cast<T>(-std::numeric_limits<float>::infinity());
                    } else {
                        y_ptr[y_curr] = utils::cast<T>(static_cast<float>(running_max + std::log(running_sum_exp)));
                    }
                }

                // Update state (Log-Add-Exp trick)
                // exp(log(sum) + log(1 + exp(val - max)))
                if (val > running_max) {
                    // 新值更大，更新 max，并缩小旧的 sum
                    running_sum_exp = running_sum_exp * std::exp(running_max - val) + 1.0;
                    running_max = val;
                } else {
                    // 新值较小，直接累加
                    running_sum_exp += std::exp(val - running_max);
                }

                // Case: Inclusive (Scan 包含当前元素)
                if (!exclusive) {
                    y_ptr[y_curr] = utils::cast<T>(static_cast<float>(running_max + std::log(running_sum_exp)));
                }
            }
        }
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    auto dtype = _info.dtype();
    // 将不透明指针传递给实现函数
    const Opaque* meta = _opaque;

    switch (dtype) {
    case INFINI_DTYPE_F32:
        cpu::calculate_cpu_impl<float>(_info, meta, y, x);
        break;
    case INFINI_DTYPE_F16:
        cpu::calculate_cpu_impl<fp16_t>(_info, meta, y, x);
        break;
    case INFINI_DTYPE_BF16:
        cpu::calculate_cpu_impl<bf16_t>(_info, meta, y, x);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::logcumsumexp::cpu