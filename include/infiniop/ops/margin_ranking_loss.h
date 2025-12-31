#ifndef __INFINIOP_MARGIN_RANKING_LOSS_API_H__
#define __INFINIOP_MARGIN_RANKING_LOSS_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopMarginRankingLossDescriptor_t;

__C __export infiniStatus_t infiniopCreateMarginRankingLossDescriptor(infiniopHandle_t handle,
                                                                      infiniopMarginRankingLossDescriptor_t *desc_ptr,
                                                                      infiniopTensorDescriptor_t output,
                                                                      infiniopTensorDescriptor_t input1,
                                                                      infiniopTensorDescriptor_t input2,
                                                                      infiniopTensorDescriptor_t target,
                                                                      float margin,
                                                                      int p, // <--- 新增参数
                                                                      int reduction);

__C __export infiniStatus_t infiniopGetMarginRankingLossWorkspaceSize(infiniopMarginRankingLossDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopMarginRankingLoss(infiniopMarginRankingLossDescriptor_t desc,
                                                      void *workspace,
                                                      size_t workspace_size,
                                                      void *output,
                                                      const void *input1,
                                                      const void *input2,
                                                      const void *target,
                                                      void *stream);

__C __export infiniStatus_t infiniopDestroyMarginRankingLossDescriptor(infiniopMarginRankingLossDescriptor_t desc);

#endif // __INFINIOP_MARGIN_RANKING_LOSS_API_H__