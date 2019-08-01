#include <ActivationFunction/SigmoidFunction.h>

void SigmoidFunction::forward_propagation(Eigen::MatrixXf &input) const {
  input.array() = (1 / (1 + Eigen::exp((-1) * input.array())));
}

Eigen::MatrixXf
SigmoidFunction::apply_derivative(const Eigen::MatrixXf &m_a,
                                  const Eigen::MatrixXf &dC_da) const {
  return ((m_a.array() * (1 - m_a.array())) * dC_da.array()).matrix();
}