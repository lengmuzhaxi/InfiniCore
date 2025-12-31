#ifndef __MARGIN_RANKING_LOSS_H__
#define __MARGIN_RANKING_LOSS_H__

#include "../../operator.h"
#include "info.h" // 引用对应的 MarginRankingLossInfo 定义

// 宏定义：用于生成不同命名空间下的 Descriptor 类
#define DESCRIPTOR(NAMESPACE)                                            \
    namespace op::margin_ranking_loss::NAMESPACE {                       \
    class Descriptor final : public InfiniopDescriptor {                 \
        struct Opaque;                                                   \
        Opaque *_opaque;                                                 \
        MarginRankingLossInfo _info;                                     \
        size_t _workspace_size;                                          \
                                                                         \
        Descriptor(                                                      \
            Opaque *opaque,                                              \
            MarginRankingLossInfo info,                                  \
            size_t workspace_size,                                       \
            infiniDevice_t device_type,                                  \
            int device_id)                                               \
            : InfiniopDescriptor{device_type, device_id},                \
              _opaque(opaque),                                           \
              _info(info),                                               \
              _workspace_size(workspace_size) {}                         \
                                                                         \
    public:                                                              \
        ~Descriptor();                                                   \
                                                                         \
        size_t workspaceSize() const { return _workspace_size; }         \
                                                                         \
        static infiniStatus_t create(                                    \
            infiniopHandle_t handle,                                     \
            Descriptor **desc_ptr,                                       \
            infiniopTensorDescriptor_t out_desc,                         \
            infiniopTensorDescriptor_t input1_desc,                      \
            infiniopTensorDescriptor_t input2_desc,                      \
            infiniopTensorDescriptor_t target_desc,                      \
            float margin,                                                \
            int p,                                                       \
            int reduction);                                              \
                                                                         \
        infiniStatus_t calculate(                                        \
            void *workspace,                                             \
            size_t workspace_size,                                       \
            void *output,                                                \
            const void *input1,                                          \
            const void *input2,                                          \
            const void *target,                                          \
            void *stream) const;                                         \
    };                                                                   \
    }

#endif // __MARGIN_RANKING_LOSS_H__