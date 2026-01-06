#include "split_nvidia.cuh"
#include "../cuda/kernel.cuh"
#include "../../../handle.h"
#include "../../../devices/nvidia/nvidia_handle.h"
namespace op::split::nvidia {
template <typename T>
void launch_impl(
    const SplitInfo &info,
    std::vector<void *> outputs,
    const void *input,
    void *stream) {

    auto in_ptr = reinterpret_cast<const T *>(input);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    int ndim = info.ndim();
    int64_t axis = info.axis();

    // 【修复】显式使用 cuda::TensorMeta
    cuda::TensorMeta in_meta;
    in_meta.ndim = ndim;
    const size_t* in_sh = info.input_shape();
    const ptrdiff_t* in_st = info.input_strides();
    
    // 【修复】显式使用 cuda::MAX_NDIM
    for(int i=0; i < cuda::MAX_NDIM; ++i) {
        if (i < ndim) {
            in_meta.shape[i] = in_sh[i];
            in_meta.strides[i] = in_st[i];
        } else {
            in_meta.shape[i] = 1;
            in_meta.strides[i] = 0;
        }
    }

    const auto& out_infos = info.outputs(); // 这里使用的是 op::split::TensorMeta
    size_t current_axis_offset = 0;

    for (size_t k = 0; k < outputs.size(); ++k) {
        auto out_ptr = reinterpret_cast<T *>(outputs[k]);
        const auto& meta_info = out_infos[k];

        // 【修复】显式使用 cuda::TensorMeta
        cuda::TensorMeta out_meta;
        out_meta.ndim = ndim;
        size_t total_elements = 1;

        // 【修复】显式使用 cuda::MAX_NDIM
        for(int i=0; i < cuda::MAX_NDIM; ++i) {
            if (i < ndim) {
                out_meta.shape[i] = meta_info.shape[i];
                out_meta.strides[i] = meta_info.strides[i];
                total_elements *= meta_info.shape[i];
            } else {
                out_meta.shape[i] = 1;
                out_meta.strides[i] = 0;
            }
        }

        if (total_elements > 0) {
            unsigned int block_size = 256;
            unsigned int grid_size = (total_elements + block_size - 1) / block_size;

            // 【修复】显式调用 cuda::split_kernel
            cuda::split_kernel<T><<<grid_size, block_size, 0, cuda_stream>>>(
                out_ptr,
                in_ptr,
                out_meta,
                in_meta,
                axis,
                current_axis_offset,
                total_elements
            );
        }

        current_axis_offset += meta_info.shape[axis];
    }
}

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    if (_opaque) delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    std::vector<infiniopTensorDescriptor_t> output_descs,
    std::vector<infiniopTensorDescriptor_t> input_descs,
    int64_t axis) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);

    if (input_descs.empty() || output_descs.empty()) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto result = SplitInfo::create(output_descs, input_descs[0], axis);
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
    std::vector<void *> outputs,
    std::vector<const void *> inputs,
    void *stream) const {

    if (inputs.empty() || outputs.size() != _info.outputs().size()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    const void *input = inputs[0];
    auto dtype = _info.dtype();

    switch (dtype) {
    case INFINI_DTYPE_F16:
        launch_impl<half>(_info, outputs, input, stream);
        break;
    case INFINI_DTYPE_BF16:
        launch_impl<nv_bfloat16>(_info, outputs, input, stream);
        break;
    case INFINI_DTYPE_F32:
        launch_impl<float>(_info, outputs, input, stream);
        break;
    case INFINI_DTYPE_F64:
        launch_impl<double>(_info, outputs, input, stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::split::nvidia