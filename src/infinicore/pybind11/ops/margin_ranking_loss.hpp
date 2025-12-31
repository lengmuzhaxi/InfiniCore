#pragma once

#include <pybind11/pybind11.h>
#include "infinicore/ops/margin_ranking_loss.hpp" 

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_margin_ranking_loss(py::module &m) {
    // 1. 绑定 functional 接口
    m.def("margin_ranking_loss",
          &op::margin_ranking_loss,
          py::arg("input1"),
          py::arg("input2"),
          py::arg("target"),
          py::arg("margin") = 0.0f,
          py::arg("p") = 1,           // <--- 新增 p 参数，默认值为 1
          py::arg("reduction") = 1,
          R"doc(Creates a criterion that measures the loss given inputs x1, x2, two 1D mini-batch Tensors, and a label 1D mini-batch tensor y (containing 1 or -1).

    Args:
        input1 (Tensor): The first input tensor.
        input2 (Tensor): The second input tensor.
        target (Tensor): The target tensor (containing 1 or -1).
        margin (float, optional): Has a default value of 0.
        p (int, optional): The norm degree. Default: 1.
        reduction (int, optional): Specifies the reduction to apply to the output: 0='none', 1='mean', 2='sum'. Default: 1.
    )doc");

    // 2. 绑定 explicit output 接口
    m.def("margin_ranking_loss_",
          &op::margin_ranking_loss_,
          py::arg("output"),
          py::arg("input1"),
          py::arg("input2"),
          py::arg("target"),
          py::arg("margin") = 0.0f,
          py::arg("p") = 1,           // <--- 新增 p 参数
          py::arg("reduction") = 1,
          R"doc(Explicit output MarginRankingLoss operation. Writes the result into the output tensor.)doc");
}

} // namespace infinicore::ops