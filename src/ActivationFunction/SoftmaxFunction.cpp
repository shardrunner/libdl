#include "ActivationFunction/SoftmaxFunction.h"
Eigen::MatrixXf
SoftmaxFunction::apply_function(const Eigen::MatrixXf &input) const {
  //TODO Check if correctly applied per column
  /*auto output=input.array().exp();
  return (output.rowwise()/output.colwise().sum()).matrix();*/

  //auto output = (input.rowwise() - input.colwise().maxCoeff()).array().exp();
  //auto colsums = output.colwise().sum();
  //output.array().rowwise() /= colsums;
  return input;

}
Eigen::MatrixXf
SoftmaxFunction::apply_derivate(const Eigen::MatrixXf &input) const {
   return Eigen::MatrixXf();

/*  auto temp = np.diag(s)

  for i in range(len(jacobian_m)):
  for j in range(len(jacobian_m)):
  if i == j:
  jacobian_m[i][j] = s[i] * (1-s[i])
  else:
  jacobian_m[i][j] = -s[i]*s[j]
  return jacobian_m*/
}
