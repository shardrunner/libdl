#include "ActivationFunction/IdentityFunction.h"
Eigen::MatrixXf
IdentityFunction::apply_function(const Eigen::MatrixXf &input) const {
  return input;
}
Eigen::MatrixXf
IdentityFunction::apply_derivative(const Eigen::MatrixXf &m_a,
                                   const Eigen::MatrixXf &dC_da) const {
  return dC_da;
}
