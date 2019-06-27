#include "Layer/MaxPoolLayer.h"
void MaxPoolLayer::feed_forward(const Eigen::MatrixXf &input) {

  // TODO catch not even

  int w_out = (input.rows() - m_w.rows()) / 2 + 1;
  int h_out = (input.cols() - m_w.cols()) / 2 + 1;
  m_z = Eigen::MatrixXf(w_out, h_out);

  for (int i = 0; i < w_out; i++) {
    for (int j = 0; j < h_out; j++) {
      m_z(i, j) = (input.block(i * 2, j * 2, m_w.rows(), m_w.cols()).array())
                      .maxCoeff() +
                  m_b(0);
    }
  }
  m_a = m_activation_function->apply_function(m_z);
}
void MaxPoolLayer::backpropagation(const Eigen::MatrixXf &a_prev,
                                   const Eigen::MatrixXf &dC_da) {}
const Eigen::MatrixXf &MaxPoolLayer::get_forward_output() { return m_a; }
const Eigen::MatrixXf &MaxPoolLayer::get_backward_output() {
  return m_dC_da_prev;
}
void MaxPoolLayer::initialize_parameter() {}
void MaxPoolLayer::update_parameter() {}

MaxPoolLayer::MaxPoolLayer(
    int number_input_channel, int number_output_channel, int filter_width,
    int filter_heigth, std::unique_ptr<ActivationFunction> activation_function,
    std::unique_ptr<RandomInitialization> random_initialization)
    : m_number_input_channel(number_input_channel),
      m_number_output_channel(number_output_channel),
      m_activation_function(std::move(activation_function)),
      m_random_initialization(std::move(random_initialization)) {
  m_w.resize(filter_heigth, filter_width);
  m_b.resize(number_output_channel);
}