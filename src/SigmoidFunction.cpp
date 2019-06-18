#include <SigmoidFunction.h>

void SigmoidFunction::apply_function(Eigen::MatrixXf &input) {
  input = (1 / (1 + Eigen::exp((-1) * input.array()))).matrix();
}

void SigmoidFunction::apply_derivate(Eigen::MatrixXf &input) {
  SigmoidFunction::apply_function(input);
  input=(input.array()*(1-input.array())).matrix();
}