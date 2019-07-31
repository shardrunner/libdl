#include "ActivationFunction/IdentityFunction.h"

void
IdentityFunction::forward_propagation(Eigen::MatrixXf &input) const {

}
Eigen::MatrixXf
IdentityFunction::backward_propagation(const Eigen::MatrixXf &m_a,
                                       const Eigen::MatrixXf &dC_da) const {
  return dC_da;
}
