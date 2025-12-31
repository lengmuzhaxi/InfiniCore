#ifndef __MARGIN_RANKING_LOSS_INFO_H__
#define __MARGIN_RANKING_LOSS_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::margin_ranking_loss {

class MarginRankingLossInfo {
    MarginRankingLossInfo() = default;

public:
    int _dtype;      // 输入/输出的数据类型
    float _margin;   // 边界值
    int _p;          // 范数次数 (根据测试用例推断需要支持 p，通常为 1 或 2)
    int _reduction;  // 规约模式 (0:None, 1:Mean, 2:Sum)

    int dtype() const { return _dtype; }
    float margin() const { return _margin; }
    int p() const { return _p; }
    int reduction() const { return _reduction; }

    // 构造函数
    MarginRankingLossInfo(int dtype, float margin, int p, int reduction)
        : _dtype(dtype), _margin(margin), _p(p), _reduction(reduction) {}

    static utils::Result<MarginRankingLossInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t input1_desc,
        infiniopTensorDescriptor_t input2_desc,
        infiniopTensorDescriptor_t target_desc,
        float margin,
        int p, // 通常 MarginRankingLoss 不带 p，但根据您的测试用例数据 ((...), p_or_None) 增加此支持
        int reduction) {

        // 1. 基础指针检查
        if (out_desc == nullptr || input1_desc == nullptr || input2_desc == nullptr || target_desc == nullptr) {
            return INFINI_STATUS_BAD_PARAM;
        }

        // 2. 检查数据类型一致性
        // MarginRankingLoss 要求 Input1, Input2, Target (1/-1) 类型一致
        int dtype = input1_desc->dtype();
        if (input2_desc->dtype() != dtype || target_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        
        // 3. 检查输出类型
        if (out_desc->dtype() != dtype) {
             return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // 4. 检查输出形状与规约模式
        if (reduction != 0) {
            // Reduction::Mean/Sum -> 输出必须是标量
            if (out_desc->numel() != 1) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        } else {
            // Reduction::None -> 输出形状由广播决定，此处不进行复杂的形状广播推导检查，
            // 但必须确保输出不为空
            if (out_desc->numel() == 0) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        // 5. 校验 p 参数 (仅支持 1 或 2，如果业务逻辑需要)
        if (p != 1 && p != 2) {
             return INFINI_STATUS_BAD_PARAM;
        }

        return utils::Result<MarginRankingLossInfo>(MarginRankingLossInfo{
            dtype,      // _dtype
            margin,     // _margin
            p,          // _p
            reduction   // _reduction
        });
    }
};

} // namespace op::margin_ranking_loss

#endif // __MARGIN_RANKING_LOSS_INFO_H__