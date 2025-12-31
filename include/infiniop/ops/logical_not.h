#ifndef __INFINIOP_LOGICAL_NOT_API_H__
#define __INFINIOP_LOGICAL_NOT_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLogicalNotDescriptor_t;

__C __export infiniStatus_t
infiniopCreateLogicalNotDescriptor(infiniopHandle_t handle,
                                   infiniopLogicalNotDescriptor_t *desc_ptr,
                                   infiniopTensorDescriptor_t output,
                                   infiniopTensorDescriptor_t input);

__C __export infiniStatus_t
infiniopGetLogicalNotWorkspaceSize(infiniopLogicalNotDescriptor_t desc,
                                   size_t *size);

__C __export infiniStatus_t
infiniopLogicalNot(infiniopLogicalNotDescriptor_t desc,
                   void *workspace,
                   size_t workspace_size,
                   void *output,
                   const void *input,
                   void *stream);

__C __export infiniStatus_t
infiniopDestroyLogicalNotDescriptor(infiniopLogicalNotDescriptor_t desc);

#endif
