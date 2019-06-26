#include "ActivationFunction/TanFunction.h"

Eigen::MatrixXf
TanFunction::apply_function(const Eigen::MatrixXf &input) const {
  return (input.array().tanh()).matrix();
}
Eigen::MatrixXf
TanFunction::apply_derivate(const Eigen::MatrixXf &m_a,
                            const Eigen::MatrixXf &dC_da) const {
  //return (1-apply_function(m_a).array().pow(2)).matrix();
  return ((1-m_a.array().pow(2))*dC_da.array()).matrix();
}
