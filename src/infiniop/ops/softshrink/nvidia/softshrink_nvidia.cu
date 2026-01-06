#include "softshrink_nvidia.cuh"
#include "../cuda/kernel.cuh"
#include "../../../handle.h"
#include "../../../devices/nvidia/nvidia_handle.h"

namespace op::softshrink::nvidia {

constexpr int MAX_NDIM = 8;

struct TensorMeta {
    int ndim;
    size_t shape[MAX_NDIM];
    ptrdiff_t strides[MAX_NDIM];
};

template <typename T>
__global__ void softshrink_kernel(
    T *output,
    const T *input,
    size_t n,
    float lambd,
    TensorMeta meta) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        size_t offset = 0;
        size_t remaining = idx;

        #pragma unroll
        for (int i = meta.ndim - 1; i >= 0; --i) {
            size_t dim_idx = remaining % meta.shape[i];
            remaining /= meta.shape[i];
            offset += dim_idx * meta.strides[i];
        }

        T val = input[offset];
        
        op::softshrink::cuda::SoftshrinkOp op;
        output[idx] = op(val, lambd);
    }
}

template <typename T>
void launch_impl(
    const SoftshrinkInfo &info,
    void *output,
    const void *input,
    void *stream) {

    size_t n = info.total_elements();
    float lambd = info.lambd();

    TensorMeta meta;
    meta.ndim = info.ndim();
    
    const size_t* sh = info.shape();
    const ptrdiff_t* st = info.strides();

    for(int i=0; i<MAX_NDIM; ++i) {
        meta.shape[i] = 1;
        meta.strides[i] = 0;
    }

    for(int i=0; i < meta.ndim && i < MAX_NDIM; ++i) {
        meta.shape[i] = sh[i];
        meta.strides[i] = st[i];
    }

    auto out_ptr = reinterpret_cast<T *>(output);
    auto in_ptr = reinterpret_cast<const T *>(input);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    unsigned int block_size = 256;
    unsigned int grid_size = (n + block_size - 1) / block_size;

    softshrink_kernel<T><<<grid_size, block_size, 0, cuda_stream>>>(
        out_ptr,
        in_ptr,
        n,
        lambd,
        meta
    );
}

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    if (_opaque) delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec,
    float lambd) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);

    if (input_desc_vec.empty()) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto result = SoftshrinkInfo::create(out_desc, input_desc_vec[0], lambd);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        new Opaque(),
        result.take(),
        0, 
        handle->device,
        handle->device_id
    );

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    if (inputs.empty()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    const void *input = inputs[0];
    auto dtype = _info.dtype();

    switch (dtype) {
    case INFINI_DTYPE_F16:
        launch_impl<half>(_info, output, input, stream);
        break;
    case INFINI_DTYPE_BF16:
        launch_impl<nv_bfloat16>(_info, output, input, stream);
        break;
    case INFINI_DTYPE_F32:
        launch_impl<float>(_info, output, input, stream);
        break;
    case INFINI_DTYPE_F64:
        launch_impl<double>(_info, output, input, stream);
        break;
    case INFINI_DTYPE_I32:
        launch_impl<int32_t>(_info, output, input, stream);
        break;
    case INFINI_DTYPE_I64:
        launch_impl<int64_t>(_info, output, input, stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::softshrink::nvidia