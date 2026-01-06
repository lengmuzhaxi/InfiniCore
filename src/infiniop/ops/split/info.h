#ifndef __SPLIT_INFO_H__
#define __SPLIT_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>
#include <cstring> // for memcpy
#include <numeric> // for accumulate

namespace op::split {

// 定义最大支持维度
constexpr int MAX_NDIM = 8;

// 辅助结构：存储单个 Tensor 的元数据 (Shape & Strides)
struct TensorMeta {
    size_t shape[MAX_NDIM];
    ptrdiff_t strides[MAX_NDIM];
};

class SplitInfo {
    SplitInfo() = default;

public:
    int _dtype;
    int _ndim;
    int64_t _axis;
    size_t _total_elements; // 输入的总元素数

    // 输入 Tensor 的形状和步长
    size_t _input_shape[MAX_NDIM];
    ptrdiff_t _input_strides[MAX_NDIM];

    // 输出 Tensor 的元数据列表
    std::vector<TensorMeta> _outputs;

    // Getters
    int dtype() const { return _dtype; }
    int ndim() const { return _ndim; }
    int64_t axis() const { return _axis; }
    size_t total_elements() const { return _total_elements; }
    const size_t* input_shape() const { return _input_shape; }
    const ptrdiff_t* input_strides() const { return _input_strides; }
    const std::vector<TensorMeta>& outputs() const { return _outputs; }

    // Constructor
    SplitInfo(int dtype, int ndim, int64_t axis, size_t total_elements,
              const std::vector<size_t>& in_shape,
              const std::vector<ptrdiff_t>& in_strides,
              std::vector<TensorMeta> outputs)
        : _dtype(dtype), _ndim(ndim), _axis(axis), _total_elements(total_elements), _outputs(std::move(outputs)) {

        // 拷贝输入 Shape 和 Strides 到定长数组
        if (_ndim > MAX_NDIM) {
             _ndim = MAX_NDIM;
        }
        std::memcpy(_input_shape, in_shape.data(), _ndim * sizeof(size_t));
        std::memcpy(_input_strides, in_strides.data(), _ndim * sizeof(ptrdiff_t));
    }

    static utils::Result<SplitInfo> create(
        const std::vector<infiniopTensorDescriptor_t>& output_descs,
        infiniopTensorDescriptor_t input_desc,
        int64_t axis) {

        int dtype = input_desc->dtype();
        int ndim = input_desc->ndim();

        // 1. 处理负数轴
        if (axis < 0) {
            axis += ndim;
        }
        if (axis < 0 || axis >= ndim) {
            return INFINI_STATUS_BAD_PARAM;
        }

        const auto& in_shape = input_desc->shape();
        const auto& in_strides = input_desc->strides();
        size_t total_elements = 1;
        
        // 准备输入的 shape/stride vector
        std::vector<size_t> in_shape_vec(ndim);
        std::vector<ptrdiff_t> in_strides_vec(ndim);

        for (int i = 0; i < ndim; ++i) {
            total_elements *= in_shape[i];
            in_shape_vec[i] = in_shape[i];
            in_strides_vec[i] = in_strides[i];
        }

        // 2. 验证输出并构建元数据
        std::vector<TensorMeta> output_metas;
        output_metas.reserve(output_descs.size());

        size_t accumulated_split_size = 0;

        for (const auto& out_desc : output_descs) {
            // 类型检查
            if (out_desc->dtype() != dtype) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
            // 【修复】添加 static_cast<int> 避免 signed/unsigned 比较警告
            if (static_cast<int>(out_desc->ndim()) != ndim) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }

            const auto& out_shape = out_desc->shape();
            const auto& out_strides = out_desc->strides();

            TensorMeta meta;
            for (int i = 0; i < ndim; ++i) {
                // 形状检查：非 axis 维度必须一致
                if (i != axis && out_shape[i] != in_shape[i]) {
                    return INFINI_STATUS_BAD_TENSOR_SHAPE;
                }
                if (i < MAX_NDIM) {
                    meta.shape[i] = out_shape[i];
                    meta.strides[i] = out_strides[i];
                }
            }
            output_metas.push_back(meta);
            accumulated_split_size += out_shape[axis];
        }
        if (accumulated_split_size != in_shape[axis]) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        return utils::Result<SplitInfo>(SplitInfo{
            dtype,
            ndim,
            axis,
            total_elements,
            in_shape_vec,
            in_strides_vec,
            std::move(output_metas)
        });
    }
};

} // namespace op::split

#endif // __SPLIT_INFO_H__