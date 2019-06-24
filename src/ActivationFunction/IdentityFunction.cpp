#include "ActivationFunction/IdentityFunction.h"
Eigen::MatrixXf
IdentityFunction::apply_function(const Eigen::MatrixXf &input) const {
  return input;
}
Eigen::MatrixXf
IdentityFunction::apply_derivate(const Eigen::MatrixXf &input) const {
  return Eigen::MatrixXf::Constant(input.rows(),input.cols(),1);
}
