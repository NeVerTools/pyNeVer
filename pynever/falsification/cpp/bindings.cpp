#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include "CPVerificationSolver.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_verify, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    py::class_<FullyConnectedLayer>(m, "FullyConnectedLayer")
        .def(py::init<std::vector<std::vector<double>>, std::vector<double>, bool>());

    py::class_<FullyConnectedNetwork>(m, "FullyConnectedNetwork")
        .def(py::init<std::vector<FullyConnectedLayer>>());

    py::class_<PropertyBounds>(m, "PropertyBounds")
        .def(py::init<std::vector<std::vector<double>>,std::vector<double>, std::vector<std::vector<double>>, std::vector<double>>());

    py::class_<CPVerificationSolver>(m, "CPVerificationSolver")
        .def(py::init<FullyConnectedNetwork, PropertyBounds>())
        .def("solve", &CPVerificationSolver::solve, py::call_guard<py::gil_scoped_release>());

}