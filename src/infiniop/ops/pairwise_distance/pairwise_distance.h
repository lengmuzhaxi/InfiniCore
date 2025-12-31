#ifndef __PAIRWISE_DISTANCE_H__
#define __PAIRWISE_DISTANCE_H__

#include "../../operator.h"
#include "info.h" // 引用对应的 PairwiseDistanceInfo 定义

// 宏定义：用于生成不同命名空间下的 Descriptor 类
#define DESCRIPTOR(NAMESPACE)                                            \
    namespace op::pairwise_distance::NAMESPACE {                         \
    class Descriptor final : public InfiniopDescriptor {                 \
        struct Opaque;                                                   \
        Opaque *_opaque;                                                 \
        PairwiseDistanceInfo _info;                                      \
        size_t _workspace_size;                                          \
                                                                         \
        Descriptor(                                                      \
            Opaque *opaque,                                              \
            PairwiseDistanceInfo info,                                   \
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
            infiniopTensorDescriptor_t output_desc,                      \
            infiniopTensorDescriptor_t x1_desc,                          \
            infiniopTensorDescriptor_t x2_desc,                          \
            float p,                                                     \
            float eps,                                                   \
            bool keepdim);                                               \
                                                                         \
        infiniStatus_t calculate(                                        \
            void *workspace,                                             \
            size_t workspace_size,                                       \
            void *output,                                                \
            const void *x1,                                              \
            const void *x2,                                              \
            void *stream) const;                                         \
    };                                                                   \
    }

#endif // __PAIRWISE_DISTANCE_H__