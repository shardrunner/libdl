#include "ActivationFunction/SoftmaxFunction.h"
Eigen::MatrixXf
SoftmaxFunction::apply_function(const Eigen::MatrixXf &input) const {
  //TODO Check if correctly applied per column
  auto output=input.array().exp();
  return (output.rowwise()/output.colwise().sum()).matrix();

}
Eigen::MatrixXf
SoftmaxFunction::apply_derivate(const Eigen::MatrixXf &input) const {
  return Eigen::MatrixXf();
}
