#pragma once

#include <pybind11/pybind11.h>
#include "infinicore/ops/pixel_shuffle.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_pixel_shuffle(py::module &m) {
    // Out-of-place: pixel_shuffle(input, upscale_factor) -> output
    m.def("pixel_shuffle",
          &op::pixel_shuffle,
          py::arg("input"),
          py::arg("upscale_factor"),
          R"doc(Rearranges elements in a tensor of shape (*, C, H, W) to a tensor of shape (*, C/r^2, H*r, W*r), where r is an upscale factor.

Args:
    input: The input tensor.
    upscale_factor: factor to increase spatial resolution by.

Returns:
    A new tensor with shuffled pixels.
)doc");

    // In-place-like: pixel_shuffle_(output, input, upscale_factor)
    m.def("pixel_shuffle_",
          &op::pixel_shuffle_,
          py::arg("output"),
          py::arg("input"),
          py::arg("upscale_factor"),
          R"doc(In-place version of pixel_shuffle (writes to output).

Args:
    output: The output tensor.
    input: The input tensor.
    upscale_factor: factor to increase spatial resolution by.
)doc");
}

} // namespace infinicore::ops