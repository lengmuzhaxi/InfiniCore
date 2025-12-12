#pragma once

#include <pybind11/pybind11.h>
#include "infinicore/ops/erfc.hpp" 
namespace py = pybind11;

namespace infinicore::ops {

inline void bind_erfc(py::module &m) {
    m.def("erfc",
          &op::erfc,
          py::arg("input"),
          R"doc(Computes the complementary error function of each element of input.

Returns a new tensor with the complementary error function of the elements of input.)doc");
    m.def("erfc_",
          &op::erfc_,
          py::arg("output"),
          py::arg("input"),
          R"doc(In-place erfc operation. Writes result into output tensor.)doc");
}

} // namespace infinicore::ops