#include <ActivationFunction/SigmoidFunction.h>

void
SigmoidFunction::apply_function(Eigen::MatrixXf &input) const {
  // Apply sigmoid function element wise
  input.array() = (1 / (1 + Eigen::exp((-1) * input.array())));
}

Eigen::MatrixXf
SigmoidFunction::apply_derivative(const Eigen::MatrixXf &m_a,
                                  const Eigen::MatrixXf &dC_da) const {
  // Apply sigmoid derivative element wise and multiply with derivative next
  // layer
  return ((m_a.array() * (1 - m_a.array())) * dC_da.array()).matrix();
}