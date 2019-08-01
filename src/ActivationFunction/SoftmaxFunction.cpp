#include "ActivationFunction/SoftmaxFunction.h"

void SoftmaxFunction::forward_propagation(Eigen::MatrixXf &input) const {
  // Subtract maximum of each column to lower numerical errors and apply exp
  auto output = (input.rowwise() - input.colwise().maxCoeff()).array().exp();

  // softmax: exp(a)/sum(exp(Matrix))
  input.array() = (output.rowwise() / output.colwise().sum());
}

Eigen::MatrixXf
SoftmaxFunction::apply_derivative(const Eigen::MatrixXf &m_a,
                                  const Eigen::MatrixXf &dC_da) const {
  // from
  // https://github.com/yixuan/MiniDNN/blob/master/include/Activation/Softmax.h
  Eigen::Array<float, 1, Eigen::Dynamic> a_dot_f =
      m_a.cwiseProduct(dC_da).colwise().sum();
  return (m_a.array() * (dC_da.array().rowwise() - a_dot_f)).matrix();

  // other softmax links
  // https://stats.stackexchange.com/questions/267576/matrix-representation-of-softmax-derivatives-in-backpropagation
  // https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
  // https://medium.com/@aerinykim/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
}