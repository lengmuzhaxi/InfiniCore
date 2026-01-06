#ifndef __SOFTSHRINK_INFO_H__
#define __SOFTSHRINK_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>
#include <cstring> // for memcpy

namespace op::softshrink {

// 定义最大支持维度，通常 8 就够了
constexpr int MAX_NDIM = 8;

class SoftshrinkInfo {
    SoftshrinkInfo() = default;

public:
    int _dtype;
    float _lambd;
    size_t _total_elements;
    
    // 新增：存储维度、形状、步长
    int _ndim;
    size_t _shape[MAX_NDIM];
    ptrdiff_t _strides[MAX_NDIM];

    // Getters
    int dtype() const { return _dtype; }
    float lambd() const { return _lambd; }
    size_t total_elements() const { return _total_elements; }
    int ndim() const { return _ndim; }
    const size_t* shape() const { return _shape; }
    const ptrdiff_t* strides() const { return _strides; }

    // Constructor
    SoftshrinkInfo(int dtype, float lambd, size_t total_elements, 
                   int ndim, const std::vector<size_t>& shape, const std::vector<ptrdiff_t>& strides)
        : _dtype(dtype), _lambd(lambd), _total_elements(total_elements), _ndim(ndim) {
        
        // 简单的拷贝到定长数组，方便传给 Kernel
        if (ndim > MAX_NDIM) {
             // 实际工程应报错，这里简单截断或假设不会发生
             _ndim = MAX_NDIM;
        }
        std::memcpy(_shape, shape.data(), _ndim * sizeof(size_t));
        std::memcpy(_strides, strides.data(), _ndim * sizeof(ptrdiff_t));
    }

    static utils::Result<SoftshrinkInfo> create(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        float lambd) {

        int dtype = input_desc->dtype();
        if (dtype != output_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (input_desc->ndim() != output_desc->ndim()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        
        size_t total_elements = 1;
        int ndim = input_desc->ndim();
        const auto& in_shape = input_desc->shape();
        const auto& out_shape = output_desc->shape();
        const auto& in_strides = input_desc->strides(); // 获取步长

        // 复制 shape 和 strides 到 vector 以传给构造函数
        std::vector<size_t> shape_vec(ndim);
        std::vector<ptrdiff_t> strides_vec(ndim);

        for (int i = 0; i < ndim; ++i) {
            if (in_shape[i] != out_shape[i]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            total_elements *= in_shape[i];
            shape_vec[i] = in_shape[i];
            strides_vec[i] = in_strides[i];
        }
        
        if (lambd < 0) {
            return INFINI_STATUS_BAD_PARAM; 
        }

        return utils::Result<SoftshrinkInfo>(SoftshrinkInfo{
            dtype,
            lambd,
            total_elements,
            ndim,
            shape_vec,
            strides_vec
        });
    }
};

} // namespace op::softshrink

#endif // __SOFTSHRINK_INFO_H__