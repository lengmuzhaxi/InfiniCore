#include "infinicore/ops/split.hpp"
#include <algorithm>

namespace infinicore::op {

common::OpDispatcher<Split::schema> &Split::dispatcher() {
    static common::OpDispatcher<Split::schema> dispatcher_;
    return dispatcher_;
};

void Split::execute(std::vector<Tensor> outputs, Tensor input, int64_t dim) {
    dispatcher().lookup(context::getDevice().getType())(outputs, input, dim);
}

std::vector<Tensor> split(Tensor input, int64_t split_size, int64_t dim) {
    int64_t dim_size = input->shape()[dim];
    std::vector<Tensor> outputs;
    
    for (int64_t offset = 0; offset < dim_size; offset += split_size) {
        int64_t current_size = std::min(split_size, dim_size - offset);
        auto shape = input->shape();
        shape[dim] = current_size;
        outputs.push_back(Tensor::empty(shape, input->dtype(), input->device()));
    }

    split_(outputs, input, dim);
    return outputs;
}

std::vector<Tensor> split(Tensor input, std::vector<int64_t> split_sections, int64_t dim) {
    std::vector<Tensor> outputs;
    outputs.reserve(split_sections.size());

    for (auto section_size : split_sections) {
        auto shape = input->shape();
        shape[dim] = section_size;
        outputs.push_back(Tensor::empty(shape, input->dtype(), input->device()));
    }

    split_(outputs, input, dim);
    return outputs;
}

void split_(std::vector<Tensor> outputs, Tensor input, int64_t dim) {
    Split::execute(outputs, input, dim);
}

} // namespace infinicore::op