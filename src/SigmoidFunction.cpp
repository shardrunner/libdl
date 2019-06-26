#include <ActivationFunction/SigmoidFunction.h>

Eigen::MatrixXf SigmoidFunction::apply_function(const Eigen::MatrixXf &input) const {
  return (1 / (1 + Eigen::exp((-1) * input.array()))).matrix();
}

Eigen::MatrixXf
SigmoidFunction::apply_derivate(const Eigen::MatrixXf &m_a,
                                const Eigen::MatrixXf &dC_da) const {
  //TODO Test sigmoid -> double apply? m_a already input?
  //Eigen::MatrixXf intermediate=SigmoidFunction::apply_function(m_a);
  return ((m_a.array()*(1-m_a.array()))*dC_da.array()).matrix();
}