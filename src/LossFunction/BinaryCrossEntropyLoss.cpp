#include "LossFunction/BinaryCrossEntropyLoss.h"

float BinaryCrossEntropyLoss::calculate_loss(
    const Eigen::MatrixXf &a_prev, const Eigen::VectorXi &label) const {

  int num_samples = label.size();

  Eigen::RowVectorXf row_label = label.transpose().cast<float>();

  auto left_side = row_label.cwiseProduct((a_prev.array().log()).matrix());
  auto right_side = ((1 - row_label.array()).matrix())
                        .cwiseProduct(((1 - a_prev.array()).log()).matrix());
  float cost = (left_side + right_side).sum();
  cost = (-1) * cost / float(num_samples);

  return cost;

  // return temp_loss.array().abs().log().sum() / temp_loss.cols();
}
void BinaryCrossEntropyLoss::backpropagate(const Eigen::MatrixXf &a_prev,
                                           const Eigen::VectorXi &label) {

  const int nobs = a_prev.cols();
  const int nvar = a_prev.rows();

  Eigen::RowVectorXi label_row = label.transpose();

  if ((label_row.cols() != nobs) || (label_row.rows() != nvar)) {
    throw std::invalid_argument("Target data have incorrect dimension");
  }

  // Eigen::RowVectorXf temp=label.cast<float>().transpose();
  // Eigen::MatrixXf temp1=(-1)*(temp.array()/a_prev.array()).matrix();

  // Eigen::MatrixXf temp2=((1-temp.array())/(1-a_prev.array())).matrix();

  // temp_loss= temp1+temp2;

  temp_loss = (label_row.array() == 0)
                  .select((float(1) - a_prev.array()).cwiseInverse(),
                          -a_prev.cwiseInverse())
                  .matrix();
}
const Eigen::MatrixXf &BinaryCrossEntropyLoss::get_backpropagate() const {
  return temp_loss;
}
