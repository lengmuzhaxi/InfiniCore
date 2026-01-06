#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include "infinicore/ops/split.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_split(py::module &m) {
    // 绑定 out-of-place 接口 1: 按固定大小切分 (split_size is int)
    m.def("split",
          // 【修复】将 op::Tensor 改为 infinicore::Tensor
          py::overload_cast<infinicore::Tensor, int64_t, int64_t>(&op::split),
          py::arg("input"),
          py::arg("split_size"),
          py::arg("dim") = 0,
          R"doc(Splits the tensor into chunks. Each chunk is a view of the original tensor.

If split_size is an integer, it will split the tensor into equally sized chunks (if possible).
The last chunk will be smaller if the tensor size along the given dimension is not divisible by split_size.)doc");

    m.def("split",
          // 【修复】将 op::Tensor 改为 infinicore::Tensor
          py::overload_cast<infinicore::Tensor, std::vector<int64_t>, int64_t>(&op::split),
          py::arg("input"),
          py::arg("split_sections"),
          py::arg("dim") = 0,
          R"doc(Splits the tensor into chunks with specified sizes based on the list of sections.)doc");

    m.def("split_",
          &op::split_,
          py::arg("outputs"),
          py::arg("input"),
          py::arg("dim") = 0,
          R"doc(In-place split operation. Writes results into the provided list of output tensors.)doc");
}

} // namespace infinicore::ops