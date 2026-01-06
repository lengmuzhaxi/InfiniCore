#ifndef __INFINIOP_SOFTSHRINK_API_H__
#define __INFINIOP_SOFTSHRINK_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSoftshrinkDescriptor_t;

__C __export infiniStatus_t infiniopCreateSoftshrinkDescriptor(infiniopHandle_t handle,
                                                              infiniopSoftshrinkDescriptor_t *desc_ptr,
                                                              infiniopTensorDescriptor_t output,
                                                              infiniopTensorDescriptor_t intput,
                                                              float lambda);

__C __export infiniStatus_t infiniopGetSoftshrinkWorkspaceSize(infiniopSoftshrinkDescriptor_t desc,
                                                              size_t *size);

__C __export infiniStatus_t infiniopSoftshrink(infiniopSoftshrinkDescriptor_t desc,
                                               void *workspace,
                                               size_t workspace_size,
                                               void *output,
                                               const void *intput,
                                               void *stream);

__C __export infiniStatus_t infiniopDestroySoftshrinkDescriptor(infiniopSoftshrinkDescriptor_t desc);

#endif
