#include <ActivationFunction/SigmoidFunction.h>

Eigen::MatrixXf
SigmoidFunction::apply_function(const Eigen::MatrixXf &input) const {
  // Apply sigmoid function element wise
  return (1 / (1 + Eigen::exp((-1) * input.array()))).matrix();
}

Eigen::MatrixXf
SigmoidFunction::apply_derivative(const Eigen::MatrixXf &m_a,
                                  const Eigen::MatrixXf &dC_da) const {
  // Apply sigmoid derivative element wise and mulitply with derivative next
  // layer
  return ((m_a.array() * (1 - m_a.array())) * dC_da.array()).matrix();
}