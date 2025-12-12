#pragma once

#include <pybind11/pybind11.h>
#include "infinicore/ops/erf.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_erf(py::module &m) {
    m.def("erf",
          &op::erf,
          py::arg("input"),
          R"doc(Computes the error function of each element of input.

Returns a new tensor with the error function of the elements of input.)doc");

    m.def("erf_",
          &op::erf_,
          py::arg("output"),
          py::arg("input"),
          R"doc(In-place erf operation. Writes result into output tensor.)doc");
}

} // namespace infinicore::ops