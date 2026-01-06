#ifndef __INFINIOP_OPS_SOFTSHRINK_H__
#define __INFINIOP_OPS_SOFTSHRINK_H__

#include "../../operator.h"
#include "info.h"
#include <vector> // 【关键】必须包含 vector 头文件

// 宏定义：用于生成不同命名空间下的 Descriptor 类
#define DESCRIPTOR(NAMESPACE)                                            \
    namespace op::softshrink::NAMESPACE {                                \
    class Descriptor final : public InfiniopDescriptor {                 \
        struct Opaque;                                                   \
        Opaque *_opaque;                                                 \
        SoftshrinkInfo _info;                                            \
        size_t _workspace_size;                                          \
                                                                         \
        Descriptor(                                                      \
            Opaque *opaque,                                              \
            SoftshrinkInfo info,                                         \
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
        /* 【关键修正】这里改为 std::vector<infiniopTensorDescriptor_t> */ \
        static infiniStatus_t create(                                    \
            infiniopHandle_t handle,                                     \
            Descriptor **desc_ptr,                                       \
            infiniopTensorDescriptor_t output,                           \
            std::vector<infiniopTensorDescriptor_t> inputs,              \
            float lambd);                                                \
                                                                         \
        /* 【同步修正】这里也要保持一致 (虽然之前已经是 vector) */           \
        infiniStatus_t calculate(                                        \
            void *workspace,                                             \
            size_t workspace_size,                                       \
            void *output,                                                \
            std::vector<const void *> inputs,                            \
            void *stream) const;                                         \
    };                                                                   \
    }

#endif // __INFINIOP_OPS_SOFTSHRINK_H__