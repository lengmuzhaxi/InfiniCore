#ifndef __INFINIOP_MATRIX_POWER_API_H__
#define __INFINIOP_MATRIX_POWER_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopMatrixPowerDescriptor_t;

__C __export infiniStatus_t infiniopCreateMatrixPowerDescriptor(infiniopHandle_t handle,
                                                                infiniopMatrixPowerDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t y_desc,
                                                                infiniopTensorDescriptor_t x_desc);

__C __export infiniStatus_t infiniopGetMatrixPowerWorkspaceSize(infiniopMatrixPowerDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopMatrixPower(infiniopMatrixPowerDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *y,
                                                void const *x,
                                                int n,
                                                void *stream);

__C __export infiniStatus_t infiniopDestroyMatrixPowerDescriptor(infiniopMatrixPowerDescriptor_t desc);

#endif