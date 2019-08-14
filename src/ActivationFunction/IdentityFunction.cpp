#include "ActivationFunction/IdentityFunction.h"

void IdentityFunction::forward_propagation([[maybe_unused]] Eigen::MatrixXf &input) const {}

Eigen::MatrixXf
IdentityFunction::apply_derivative([[maybe_unused]] const Eigen::MatrixXf &m_a,
                                   const Eigen::MatrixXf &dC_da) const {
  return dC_da;
}
