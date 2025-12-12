#pragma once

#include <pybind11/pybind11.h>
#include "infinicore/ops/erfinv.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_erfinv(py::module &m) {
    m.def("erfinv",
          &op::erfinv,
          py::arg("input"),
          R"doc(Computes the inverse error function of each element of input.
The domain of the inverse error function is (-1, 1).

Returns a new tensor with the inverse error function of the elements of input.)doc");

    m.def("erfinv_",
          &op::erfinv_,
          py::arg("output"),
          py::arg("input"),
          R"doc(In-place erfinv operation. Writes result into output tensor.)doc");
}

} // namespace infinicore::ops