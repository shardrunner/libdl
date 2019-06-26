#include "LossFunction/MultiClassLoss.h"

#include <iostream>

float MultiClassLoss::calculate_loss(const Eigen::MatrixXf &a_prev, const Eigen::VectorXi &label) const {
  // L = -sum(log(phat) * y)
  // in = phat
  // d(L) / d(in) = -y / phat
  // m_din contains 0 if y = 0, and -1/phat if y = 1
/*  float res = float(0);
  const int nelem = temp_loss.size();
  const float* din_data = temp_loss.data();

  for (int i = 0; i < nelem; i++)
  {
    if (din_data[i] < float(0))
    {
      res += std::log(-din_data[i]);
    }
  }

  return res / temp_loss.cols();*/

  int a = a_prev.rows();
  int b = a_prev.cols();
  int c = label.size();

  //std::cout << "\nLabel\n" << label << "\nprev\n" << a_prev << std::endl;

  float error=0.0;
  for (int i=0; i<label.size(); i++) {
    float temp=a_prev(label(i),i);
    error-=std::log(temp);
  }

  return error/label.size();



}
void MultiClassLoss::backpropagate(const Eigen::MatrixXf &a_prev,
                                   const Eigen::VectorXi &label) {
  std::cout << "\nLoss evaluation: \nA_prev:\n" << a_prev << "\nlabel\n" <<label.transpose()<<std::endl;

  // Check dimension
  const int nobs = a_prev.cols();
  const int nclass = a_prev.rows();

  if (label.size() != nobs)
  {
    throw std::invalid_argument("[class MultiClassEntropy]: Target data have incorrect dimension");
  }

  // Compute the derivative of the input of this layer
  // L = -log(phat[y])
  // in = phat
  // d(L) / d(in) = [0, 0, ..., -1/phat[y], 0, ..., 0]
  temp_loss.resize(nclass, nobs);
  temp_loss.setZero();

  for (int i = 0; i < nobs; i++)
  {
    temp_loss(label(i), i) = float(-1.0) / a_prev.coeff(label(i), i);
  }

  std::cout << "Loss derivative:\n" << temp_loss << std::endl;
}
const Eigen::MatrixXf &MultiClassLoss::get_backpropagate() const {
  return temp_loss;
}
