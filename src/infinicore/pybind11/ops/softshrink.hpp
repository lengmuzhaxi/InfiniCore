#pragma once

#include <pybind11/pybind11.h>
#include "infinicore/ops/softshrink.hpp" 

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_softshrink(py::module &m) {
    // 1. 绑定 functional 接口: output = softshrink(input, lambd)
    m.def("softshrink",
          &op::softshrink,
          py::arg("input"),
          py::arg("lambd") = 0.5f,
          R"doc(Applies the soft shrinkage function element-wise:
    SoftShrinkage(x) = x - lambda, if x > lambda
                     = x + lambda, if x < -lambda
                     = 0, otherwise

    Args:
        input (Tensor): The input tensor.
        lambd (float): The lambda value (must be no less than zero). Default: 0.5.
    )doc");

    // 2. 绑定 explicit output 接口: softshrink_(output, input, lambd)
    m.def("softshrink_",
          &op::softshrink_,
          py::arg("output"),
          py::arg("input"),
          py::arg("lambd") = 0.5f,
          R"doc(Explicit output Softshrink operation. Writes results into the output tensor.)doc");
}

} // namespace infinicore::ops