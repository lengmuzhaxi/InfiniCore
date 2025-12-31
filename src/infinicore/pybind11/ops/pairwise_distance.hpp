#pragma once

#include <pybind11/pybind11.h>
#include "infinicore/ops/pairwise_distance.hpp" 

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_pairwise_distance(py::module &m) {
    // 1. 绑定 functional 接口: output = pairwise_distance(x1, x2, ..., keepdim=False)
    m.def("pairwise_distance",
          &op::pairwise_distance,
          py::arg("x1"),
          py::arg("x2"),
          py::arg("p") = 2.0f,
          py::arg("eps") = 1e-6f,
          py::arg("keepdim") = false,
          R"doc(Computes the pairwise distance between vectors using the p-norm.

    Args:
        x1 (Tensor): First input tensor.
        x2 (Tensor): Second input tensor.
        p (float): The norm degree. Default: 2.0.
        eps (float): Small constant to avoid division by zero. Default: 1e-6.
        keepdim (bool): Whether to keep the vector dimension. Default: False.
    )doc");

    // 2. 绑定 explicit output 接口: pairwise_distance_(output, x1, x2, ..., keepdim=False)
    m.def("pairwise_distance_",
          &op::pairwise_distance_,
          py::arg("output"),
          py::arg("x1"),
          py::arg("x2"),
          py::arg("p") = 2.0f,
          py::arg("eps") = 1e-6f,
          py::arg("keepdim") = false, 
          R"doc(Explicit output PairwiseDistance operation. Writes the result into the output tensor.)doc");
}

} // namespace infinicore::ops