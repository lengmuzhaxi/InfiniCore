#pragma once

#include <pybind11/pybind11.h>
#include "infinicore/ops/matrix_power.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_matrix_power(py::module &m) {
    // Out-of-place: matrix_power(input, n) -> output
    m.def("matrix_power",
          &op::matrix_power,
          py::arg("input"),
          py::arg("n"),
          R"doc(Computes the n-th power of a square matrix for an integer n.

Args:
    input: The input tensor (must be a square matrix or batches of square matrices).
    n: The exponent.

Returns:
    A new tensor containing the result of the matrix power operation.
)doc");
    m.def("matrix_power_",
          &op::matrix_power_,
          py::arg("output"),
          py::arg("input"),
          py::arg("n"),
          R"doc(In-place matrix power operation.

Args:
    output: The output tensor to write the result to.
    input: The input tensor.
    n: The exponent.
)doc");
}

} // namespace infinicore::ops