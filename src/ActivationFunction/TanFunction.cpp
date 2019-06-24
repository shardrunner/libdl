#include "ActivationFunction/TanFunction.h"

Eigen::MatrixXf
TanFunction::apply_function(const Eigen::MatrixXf &input) const {
  return (input.array().tanh()).matrix();
}
Eigen::MatrixXf
TanFunction::apply_derivate(const Eigen::MatrixXf &input) const {
  return (1-apply_function(input).array().pow(2)).matrix();
}
