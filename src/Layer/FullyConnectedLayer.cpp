#include "Layer/FullyConnectedLayer.h"

#include <iostream>

FullyConnectedLayer::FullyConnectedLayer(
    const int input_size, const int output_size,
    std::unique_ptr<ActivationFunction> activation_function,
    std::unique_ptr<RandomInitialization> random_initialization,
    std::unique_ptr<OptimizationFunction> optimization_function)
    : m_activation_function(std::move(activation_function)),
      m_random_initialization(std::move(random_initialization)),
      m_optimization_function(std::move(optimization_function)) {
  this->m_input_size = input_size;
  this->m_output_size = output_size;
  m_w.resize(input_size, output_size);
  m_b.resize(output_size);
  initialize_parameter();
}

void FullyConnectedLayer::feed_forward(const Eigen::MatrixXf &input) {
  assert(m_w.rows() == input.rows() &&
         "FC feed forward weights and input dimensions do not match");
  assert(m_w.cols() == m_output_size &&
         "The given output size odes not match the weight size");

  m_a.noalias() = (m_w.transpose() * input).colwise() + m_b;

  m_activation_function->forward_propagation(m_a);
}

void FullyConnectedLayer::backpropagation(const Eigen::MatrixXf &a_prev,
                                          const Eigen::MatrixXf &dC_da) {
  const long number_training_samples = a_prev.cols();

  // calculate intermediate value dC/dz
  Eigen::MatrixXf dC_dz = m_activation_function->apply_derivative(m_a, dC_da);

  // normalize sum over changes/derivatives from all samples, by dividing by
  // number of samples
  m_dC_dw.noalias() = a_prev * dC_dz.transpose() / number_training_samples;

  // also normalize sum of bias changes calculating mean and shrinking to vector
  m_dC_db = dC_dz.rowwise().mean();

  // calculate derivative for prev layer (next step in backprop)
  m_dC_da_prev.noalias() = m_w * dC_dz;
}

void FullyConnectedLayer::initialize_parameter() {
  m_random_initialization->initialize(m_w);
  m_b.setZero();
}

void FullyConnectedLayer::update_parameter() {
  m_optimization_function->optimize_weights(m_w, m_dC_dw);
  m_optimization_function->optimize_bias(m_b, m_dC_db);
}

const Eigen::MatrixXf &FullyConnectedLayer::get_forward_output() const {
  return m_a;
}

const Eigen::MatrixXf &FullyConnectedLayer::get_backward_output() const {
  return m_dC_da_prev;
}

int FullyConnectedLayer::get_number_inputs() const { return m_input_size; }

int FullyConnectedLayer::get_number_outputs() const { return m_output_size; }

void FullyConnectedLayer::set_weights(const Eigen::MatrixXf &new_weights) {
  if (new_weights.rows() != m_w.rows() || new_weights.cols() != m_w.cols()) {
    throw std::invalid_argument(
        "New weights size does not match the old weight size");
  }
  m_w = new_weights;
}

void FullyConnectedLayer::set_bias(const Eigen::VectorXf &new_bias) {
  if (new_bias.rows() != m_b.rows() || new_bias.cols() != m_b.cols()) {
    throw std::invalid_argument(
        "New bias size does not match the old bias size");
  }
  m_b = new_bias;
}

const Eigen::MatrixXf &FullyConnectedLayer::get_weights() const { return m_w; }

const Eigen::VectorXf &FullyConnectedLayer::get_bias() const { return m_b; }
