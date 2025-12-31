#ifndef __INFINIOP_PAIRWISE_DISTANCE_API_H__
#define __INFINIOP_PAIRWISE_DISTANCE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopPairwiseDistanceDescriptor_t;

__C __export infiniStatus_t infiniopCreatePairwiseDistanceDescriptor(
    infiniopHandle_t handle,
    infiniopPairwiseDistanceDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t x1,
    infiniopTensorDescriptor_t x2,
    float p,
    float eps,
    bool keepdim
);

__C __export infiniStatus_t infiniopGetPairwiseDistanceWorkspaceSize(
    infiniopPairwiseDistanceDescriptor_t desc, 
    size_t *size
);

__C __export infiniStatus_t infiniopPairwiseDistance(
    infiniopPairwiseDistanceDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *x1,
    const void *x2,
    void *stream
);

__C __export infiniStatus_t infiniopDestroyPairwiseDistanceDescriptor(
    infiniopPairwiseDistanceDescriptor_t desc
);

#endif // __INFINIOP_PAIRWISE_DISTANCE_API_H__