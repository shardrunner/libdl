#include "ActivationFunction/TanhFunction.h"

void
TanhFunction::apply_function(Eigen::MatrixXf &input) const {
  // Apply tanh element wise
  input.array() = input.array().tanh();
}
Eigen::MatrixXf
TanhFunction::apply_derivative(const Eigen::MatrixXf &m_a,
                               const Eigen::MatrixXf &dC_da) const {
  // Apply tanh derivative element wise and multiply with derivative next layer
  return ((1 - m_a.array().pow(2)) * dC_da.array()).matrix();
}
