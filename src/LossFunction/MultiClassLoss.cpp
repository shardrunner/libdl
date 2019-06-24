#include "LossFunction/MultiClassLoss.h"
float MultiClassLoss::calculate_loss() const {
  float res = 0.0;
  const int nelem = temp_loss.size();
  const float* din_data = temp_loss.data();

  for (int i = 0; i < nelem; i++)
  {
    if (din_data[i] < float(0))
    {
      res += std::log(-din_data[i]);
    }
  }

  return res / temp_loss.cols();
}
void MultiClassLoss::backpropagate(const Eigen::MatrixXf &a_prev,
                                   const Eigen::MatrixXf &label) {
  // Check dimension
  const int nobs = a_prev.cols();
  const int nclass = a_prev.rows();

  if ((label.cols() != nobs) || (label.rows() != nclass))
  {
    throw std::invalid_argument("Target data have incorrect dimension");
  }

  temp_loss.resize(nclass, nobs);
  temp_loss.noalias() = -label.cwiseQuotient(a_prev);
}
const Eigen::MatrixXf &MultiClassLoss::get_backpropagate() const {
  return temp_loss;
}
