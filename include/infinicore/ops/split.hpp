#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <vector>

namespace infinicore::op {
class Split {
public:
    using schema = void (*)(std::vector<Tensor>, Tensor, int64_t);
    static void execute(std::vector<Tensor> outputs, Tensor input, int64_t dim);
    static common::OpDispatcher<schema> &dispatcher();
};

std::vector<Tensor> split(Tensor input, int64_t split_size, int64_t dim = 0);
std::vector<Tensor> split(Tensor input, std::vector<int64_t> split_sections, int64_t dim = 0);

void split_(std::vector<Tensor> outputs, Tensor input, int64_t dim);
} // namespace infinicore::op