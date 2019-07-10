#include "ActivationFunction/TanhFunction.h"

Eigen::MatrixXf
TanhFunction::apply_function(const Eigen::MatrixXf &input) const {
  // Apply tanh element wise
  return (input.array().tanh()).matrix();
}
Eigen::MatrixXf
TanhFunction::apply_derivative(const Eigen::MatrixXf &m_a,
                               const Eigen::MatrixXf &dC_da) const {
  // Apply tanh derivative element wise and multiply with derivative next layer
  return ((1 - m_a.array().pow(2)) * dC_da.array()).matrix();
}
