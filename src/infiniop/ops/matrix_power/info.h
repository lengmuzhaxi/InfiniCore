#ifndef __MATRIX_POWER_INFO_H__
#define __MATRIX_POWER_INFO_H__

#include "../../tensor.h"
#include <vector>
#include <cstdint>
#include <numeric>

namespace op::matrix_power {

struct MatrixPowerInfo {
    int _dtype;
    size_t _ndim;
    std::vector<int64_t> _shape;
    std::vector<int64_t> _in_strides;
    std::vector<int64_t> _out_strides;
    size_t _m;
    size_t _batch_size; // 展平后的 Batch Size

    // [关键修复] 初始化所有基础类型，避免脏数据
    MatrixPowerInfo() : _dtype(0), _ndim(0), _m(0), _batch_size(0) {}

    MatrixPowerInfo(int dtype, size_t ndim, 
                    std::vector<int64_t> shape,
                    std::vector<int64_t> in_strides,
                    std::vector<int64_t> out_strides,
                    size_t m)
        : _dtype(dtype), _ndim(ndim), _shape(std::move(shape)),
          _in_strides(std::move(in_strides)), _out_strides(std::move(out_strides)), _m(m) {
        
        // 计算展平的 batch_size (用于 GPU 显存分配)
        _batch_size = 1;
        if (_ndim > 2) {
            for(size_t i = 0; i < _ndim - 2; ++i) {
                _batch_size *= _shape[i];
            }
        }
    }

    int dtype() const { return _dtype; }
    size_t ndim() const { return _ndim; }
    size_t m() const { return _m; }
    size_t batch_size() const { return _batch_size; }
    
    const std::vector<int64_t>& shape() const { return _shape; }
    const std::vector<int64_t>& in_strides() const { return _in_strides; }
    const std::vector<int64_t>& out_strides() const { return _out_strides; }

    static infiniStatus_t create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t in_desc,
        MatrixPowerInfo* out_info) {

        if (out_desc->dtype() != in_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        size_t ndim = in_desc->ndim();
        if (ndim < 2 || out_desc->ndim() != ndim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        const auto &in_shape = in_desc->shape();
        const auto &out_shape = out_desc->shape();

        // 检查形状
        for (size_t i = 0; i < ndim; ++i) {
            if (in_shape[i] != out_shape[i]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        // 检查方阵
        size_t rows = in_shape[ndim - 2];
        size_t cols = in_shape[ndim - 1];
        if (rows != cols) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // 复制完整信息
        std::vector<int64_t> shape_vec(ndim);
        std::vector<int64_t> in_strides(ndim);
        std::vector<int64_t> out_strides(ndim);
        
        for(size_t i=0; i<ndim; ++i) {
            shape_vec[i] = in_shape[i];
            in_strides[i] = in_desc->stride(i);
            out_strides[i] = out_desc->stride(i);
        }

        *out_info = MatrixPowerInfo(
            in_desc->dtype(),
            ndim,
            shape_vec,
            in_strides,
            out_strides,
            rows
        );
        
        return INFINI_STATUS_SUCCESS;
    }
};

} // namespace op::matrix_power

#endif // __MATRIX_POWER_INFO_H__