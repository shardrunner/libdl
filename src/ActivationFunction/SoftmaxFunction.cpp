#include "ActivationFunction/SoftmaxFunction.h"

//#include <iostream>
// https://github.com/yixuan/MiniDNN/blob/master/include/Activation/Softmax.h
void
SoftmaxFunction::apply_function(Eigen::MatrixXf &input) const {
  // Subtract maximum of each column to lower numerical errors and apply exp
  auto output=(input.rowwise()-input.colwise().maxCoeff()).array().exp();

  input.array() = (output.rowwise()/output.colwise().sum());

  //auto A = (input.rowwise() - input.colwise().maxCoeff()).array().exp().matrix();
 // Eigen::Array<float, 1, Eigen::Dynamic> colsums = A.colwise().sum();
  //input = A.array().rowwise() / colsums;
}

Eigen::MatrixXf
SoftmaxFunction::apply_derivative(const Eigen::MatrixXf &m_a,
                                  const Eigen::MatrixXf &dC_da) const {
  //TODO not necessary to apply softmax again?
  //Eigen::MatrixXf m_a_der=m_a;
  //apply_function(m_a_der);
  //Eigen::MatrixXf output(m_a.rows(), m_a.cols());

  // Eigen::Array<float, 1, Eigen::Dynamic>
  // temp=softmax_input.cwiseProduct(input).colwise().sum();
  // output=softmax_input.array()*(input.array().rowwise() - temp);

  //auto F = dC_da;
  //auto A = m_a;

  Eigen::Array<float, 1, Eigen::Dynamic> a_dot_f =
          m_a.cwiseProduct(dC_da).colwise().sum();
  return (m_a.array() * (dC_da.array().rowwise() - a_dot_f)).matrix();

  //return output;

  // RowArray a_dot_f = A.cwiseProduct(F).colwise().sum();
  // G.array() = A.array() * (F.array().rowwise() - a_dot_f);

  // TODO Test other code

  /*  for (int i=0; i< softmax_input.rows();i++) {
      for (int j=0; j< softmax_input.cols(); j++) {
        if (i==j) {
          output(i, j) = softmax_input(i) * (1 - softmax_input(i));
        }
          else {
            output(i,j) = -softmax_input(i)*softmax_input(j);
          }
        }
      }
    return output.colwise().sum();*/

  /*
   *     # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix
   multiplication s = softmax.reshape(-1,1) return np.diagflat(s) - np.dot(s,
   s.T)
   *
   *
   * auto temp = np.diag(s)

    for i in range(len(jacobian_m)):
    for j in range(len(jacobian_m)):
    if i == j:
    jacobian_m[i][j] = s[i] * (1-s[i])
    else:
    jacobian_m[i][j] = -s[i]*s[j]
    return jacobian_m*/
}