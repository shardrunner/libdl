#include <pybind11/pybind11.h>
#include "NeuralNetwork.h"
#include "pybind11/stl.h"


#include <pybind11/eigen.h>

//http://people.duke.edu/~ccc14/sta-663-2018/notebooks/S13C_pybind11.html

namespace py = pybind11;

// https://pybind11.readthedocs.io/en/master/advanced/cast/eigen.html
//
PYBIND11_MODULE(MlibWrapper, m) {
    m.doc() = "Wrapper for Mlib"; // optional module docstring

    py::class_<NeuralNetwork>(m, "NeuralNetwork")
            .def(py::init())
            .def("add_conv_layer", &NeuralNetwork::add_conv_layer)
            .def("add_multiclass_loss", &NeuralNetwork::use_multiclass_loss)
            .def("add_fc_layer", &NeuralNetwork::add_fc_layer)
            .def("add_output_layer", &NeuralNetwork::add_output_layer)
            .def("train_network", &NeuralNetwork::train_network)
            .def("test_network", &NeuralNetwork::test_network,py::return_value_policy::reference)
            .def("train_batch", &NeuralNetwork::train_batch)
            .def("__repr__",
                 []() {
                     return "A neural Network";
                 });
}
