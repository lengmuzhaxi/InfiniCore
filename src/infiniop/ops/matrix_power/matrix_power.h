#ifndef __MATRIX_POWER_H__
#define __MATRIX_POWER_H__

#include "../../operator.h"
#include "info.h" 

#define DESCRIPTOR(NAMESPACE)                                                    \
                                                                                 \
    namespace op::matrix_power::NAMESPACE {                                      \
    class Descriptor final : public InfiniopDescriptor {                         \
        struct Opaque;                                                           \
        Opaque *_opaque;                                                         \
        MatrixPowerInfo _info;                                                   \
        size_t _workspace_size;                                                  \
                                                                                 \
        Descriptor(                                                              \
            Opaque *opaque,                                                      \
            MatrixPowerInfo info,                                                \
            size_t workspace_size,                                               \
            infiniDevice_t device_type,                                          \
            int device_id)                                                       \
            : InfiniopDescriptor{device_type, device_id},                        \
              _opaque(opaque),                                                   \
              _info(info),                                                       \
              _workspace_size(workspace_size) {}                                 \
                                                                                 \
    public:                                                                      \
        ~Descriptor();                                                           \
                                                                                 \
        size_t workspaceSize() const { return _workspace_size; }                 \
                                                                                 \
        static infiniStatus_t create(                                            \
            infiniopHandle_t handle,                                             \
            Descriptor **desc_ptr,                                               \
            infiniopTensorDescriptor_t out_desc,                                 \
            infiniopTensorDescriptor_t in_desc); /* [修改] 移除了 n */            \
                                                                                 \
        infiniStatus_t calculate(                                                \
            void *workspace,                                                     \
            size_t workspace_size,                                               \
            void *output,                                                        \
            const void *input,                                                   \
            int n, /* [修改] 添加 n */                                            \
            void *stream) const;                                                 \
    };                                                                           \
    }

#endif // __MATRIX_POWER_H__