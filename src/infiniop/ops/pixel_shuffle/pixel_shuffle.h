#ifndef __PIXEL_SHUFFLE_H__
#define __PIXEL_SHUFFLE_H__

#include "../../operator.h"
#include "info.h" // 引用 PixelShuffleInfo 定义

#define DESCRIPTOR(NAMESPACE)                                                    \
                                                                                 \
    namespace op::pixel_shuffle::NAMESPACE {                                     \
    class Descriptor final : public InfiniopDescriptor {                         \
        struct Opaque;                                                           \
        Opaque *_opaque;                                                         \
        PixelShuffleInfo _info;                                                  \
        size_t _workspace_size;                                                  \
                                                                                 \
        Descriptor(                                                              \
            Opaque *opaque,                                                      \
            PixelShuffleInfo info,                                               \
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
            infiniopTensorDescriptor_t in_desc,                                  \
            int64_t upscale_factor);                                             \
                                                                                 \
        infiniStatus_t calculate(                                                \
            void *workspace,                                                     \
            size_t workspace_size,                                               \
            void *output,                                                        \
            const void *input,                                                   \
            void *stream) const;                                                 \
    };                                                                           \
    }

#endif // __PIXEL_SHUFFLE_H__