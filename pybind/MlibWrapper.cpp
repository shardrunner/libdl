#include <pybind11/pybind11.h>

#include "NeuralNetwork.h"
#include "pybind11/iostream.h"

#include <pybind11/eigen.h>
#include "pybind11/stl.h"


namespace py = pybind11;

// https://pybind11.readthedocs.io/en/master/advanced/cast/eigen.html
/**
 * The bindings for the pybind module.
 */
PYBIND11_MODULE(MlibWrapper, m) {
    m.doc() = "Wrapper for Mlib"; // optional module docstring

    py::class_<NeuralNetwork>(m, "NeuralNetwork")
            .def(py::init())
            .def("add_conv_layer", &NeuralNetwork::add_conv_layer)
            .def("add_conv_layer_simple", &NeuralNetwork::add_conv_layer_simple)
            .def("use_multiclass_loss", &NeuralNetwork::use_multiclass_loss)
            .def("add_fc_layer", &NeuralNetwork::add_fc_layer)
            .def("add_fc_layer_relu", &NeuralNetwork::add_fc_layer_relu)
            .def("add_output_layer", &NeuralNetwork::add_output_layer)
            .def("add_output_layer_simple", &NeuralNetwork::add_output_layer_simple)
            .def("train_network", &NeuralNetwork::train_network)
            .def("test_network", &NeuralNetwork::test_network,py::return_value_policy::reference)
            .def("train_batch", &NeuralNetwork::train_batch) //,py::arg().noconvert()
            .def("set_layer_weights", &NeuralNetwork::set_layer_weights)
            .def("get_layer_weights", &NeuralNetwork::get_layer_weights)
            .def("set_layer_bias", &NeuralNetwork::set_layer_bias)
            .def("get_layer_bias", &NeuralNetwork::get_layer_bias)
            .def("check_network", &NeuralNetwork::check_network)
            .def("feed_forward_py", &NeuralNetwork::feed_forward_py)
            .def("get_current_accuracy", &NeuralNetwork::get_current_accuracy)
            .def("get_current_error", &NeuralNetwork::get_current_error)
            .def("get_current_prediction", &NeuralNetwork::get_current_prediction,py::return_value_policy::reference)
            .def("layer_size", &NeuralNetwork::layer_size,py::call_guard<py::scoped_ostream_redirect ,py::scoped_estream_redirect>())
            .def("get_predicted_classes", &NeuralNetwork::get_predicted_classes); //,py::arg().noconvert();
}