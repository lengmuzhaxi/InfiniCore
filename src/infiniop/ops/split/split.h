#ifndef __INFINIOP_OPS_SPLIT_H__
#define __INFINIOP_OPS_SPLIT_H__

#include "../../operator.h"
#include "info.h"
#include <vector>

// 宏定义：用于生成不同命名空间下的 Descriptor 类
#define DESCRIPTOR(NAMESPACE)                                            \
    namespace op::split::NAMESPACE {                                     \
    class Descriptor final : public InfiniopDescriptor {                 \
        struct Opaque;                                                   \
        Opaque *_opaque;                                                 \
        SplitInfo _info;                                                 \
        size_t _workspace_size;                                          \
                                                                         \
        Descriptor(                                                      \
            Opaque *opaque,                                              \
            SplitInfo info,                                              \
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
        /* 【新增】添加此辅助函数，供 operator.cc 调用 */                   \
        size_t numOutputs() const { return _info.outputs().size(); }     \
                                                                         \
        static infiniStatus_t create(                                    \
            infiniopHandle_t handle,                                     \
            Descriptor **desc_ptr,                                       \
            std::vector<infiniopTensorDescriptor_t> outputs,             \
            std::vector<infiniopTensorDescriptor_t> inputs,              \
            int64_t axis);                                               \
                                                                         \
        infiniStatus_t calculate(                                        \
            void *workspace,                                             \
            size_t workspace_size,                                       \
            std::vector<void *> outputs,                                 \
            std::vector<const void *> inputs,                            \
            void *stream) const;                                         \
    };                                                                   \
    }

#endif // __INFINIOP_OPS_SPLIT_H__