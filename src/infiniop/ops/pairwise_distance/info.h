#ifndef __PAIRWISE_DISTANCE_INFO_H__
#define __PAIRWISE_DISTANCE_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::pairwise_distance {

class PairwiseDistanceInfo {
    PairwiseDistanceInfo() = default;

public:
    int _dtype;              // 数据类型
    float _p;                // 范数 (float, e.g. 2.0)
    float _eps;              // 数值稳定性常数
    bool _keepdim;           // 是否保持维度

    // 形状信息缓存 (假设处理 (N, D) 或 (..., D) 结构)
    size_t _batch_size;      // N (除最后一维外的所有维度乘积)
    size_t _feature_dim;     // D (最后一维大小)

    int dtype() const { return _dtype; }
    float p() const { return _p; }
    float eps() const { return _eps; }
    bool keepdim() const { return _keepdim; }
    size_t batch_size() const { return _batch_size; }
    size_t feature_dim() const { return _feature_dim; }

    // 构造函数
    PairwiseDistanceInfo(int dtype, float p, float eps, bool keepdim, 
                         size_t batch, size_t feature_dim)
        : _dtype(dtype), _p(p), _eps(eps), _keepdim(keepdim),
          _batch_size(batch), _feature_dim(feature_dim) {}

    static utils::Result<PairwiseDistanceInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t x1_desc,
        infiniopTensorDescriptor_t x2_desc,
        float p,
        float eps,
        bool keepdim) {

        // 1. 检查输入形状一致性
        // x1 和 x2 形状必须完全一致 (暂不支持 Broadcasting，或由上层处理)
        if (x1_desc->ndim() != x2_desc->ndim()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        size_t ndim = x1_desc->ndim();
        for (size_t i = 0; i < ndim; ++i) {
            if (x1_desc->shape()[i] != x2_desc->shape()[i]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        // 2. 检查数据类型
        int dtype = x1_desc->dtype();
        if (x2_desc->dtype() != dtype || out_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // 3. 计算 N 和 D
        // Pairwise Distance 通常在最后一个维度 (D) 上进行计算
        size_t N = 1;
        size_t D = 1;

        if (ndim > 0) {
            D = x1_desc->shape()[ndim - 1]; // 最后一维是 Feature Dim
            N = x1_desc->numel() / D;       // 其余维度的乘积是 Batch
        } else {
            // 标量情况
            N = 1;
            D = 1;
        }

        // 4. 检查输出形状
        if (keepdim) {
            // Keepdim=True: 输出形状应为 (..., 1)
            // 1. 维度数必须相同
            if (out_desc->ndim() != ndim) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            // 2. 前面所有维度必须匹配
            for (size_t i = 0; i < ndim - 1; ++i) {
                if (out_desc->shape()[i] != x1_desc->shape()[i]) {
                    return INFINI_STATUS_BAD_TENSOR_SHAPE;
                }
            }
            // 3. 最后一维必须是 1
            if (out_desc->shape()[ndim - 1] != 1) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        } else {
            // Keepdim=False: 输出形状应为 (...)，即去掉最后一维
            // 1. 维度数必须少 1 (除非输入本身是 1D，输出变为标量/0D 或 1D(1) 具体取决于定义，这里假设变少)
            size_t expected_out_ndim = (ndim > 0) ? ndim - 1 : 0;
            
            if (out_desc->ndim() != expected_out_ndim) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            
            // 2. 所有剩余维度必须匹配
            for (size_t i = 0; i < expected_out_ndim; ++i) {
                if (out_desc->shape()[i] != x1_desc->shape()[i]) {
                    return INFINI_STATUS_BAD_TENSOR_SHAPE;
                }
            }
        }

        return utils::Result<PairwiseDistanceInfo>(PairwiseDistanceInfo{
            dtype,
            p,
            eps,
            keepdim,
            N,
            D
        });
    }
};

} // namespace op::pairwise_distance

#endif // __PAIRWISE_DISTANCE_INFO_H__