#include <ActivationFunction/SigmoidFunction.h>

Eigen::MatrixXf SigmoidFunction::apply_function(const Eigen::MatrixXf &input) const {
  return (1 / (1 + Eigen::exp((-1) * input.array()))).matrix();
}

Eigen::MatrixXf SigmoidFunction::apply_derivate(const Eigen::MatrixXf &input) const {
  Eigen::MatrixXf intermediate=SigmoidFunction::apply_function(input);
  return (intermediate.array()*(1-intermediate.array())).matrix();
}