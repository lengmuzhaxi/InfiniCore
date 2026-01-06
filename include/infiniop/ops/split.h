#ifndef __INFINIOP_SPLIT_API_H__
#define __INFINIOP_SPLIT_API_H__

#include "../operator_descriptor.h"
#include <stdint.h> 

typedef struct InfiniopDescriptor *infiniopSplitDescriptor_t;

__C __export infiniStatus_t infiniopCreateSplitDescriptor(infiniopHandle_t handle,
                                                          infiniopSplitDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t const *ys, 
                                                          uint64_t num_outputs,
                                                          infiniopTensorDescriptor_t x,
                                                          int64_t axis);

__C __export infiniStatus_t infiniopGetSplitWorkspaceSize(infiniopSplitDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopSplit(infiniopSplitDescriptor_t desc,
                                          void *workspace,
                                          size_t workspace_size,
                                          void *const *ys,
                                          const void *x,
                                          void *stream);

__C __export infiniStatus_t infiniopDestroySplitDescriptor(infiniopSplitDescriptor_t desc);

#endif