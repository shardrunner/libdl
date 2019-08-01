#include "LossFunction/BinaryCrossEntropyLoss.h"

double
BinaryCrossEntropyLoss::calculate_loss(const Eigen::MatrixXf &a_prev,
                                       const Eigen::VectorXi &label) const {

  const long num_samples = label.size();

  Eigen::RowVectorXf row_label = label.transpose().cast<float>();

  auto left_side = row_label.cwiseProduct((a_prev.array().log()).matrix());
  auto right_side = ((1 - row_label.array()).matrix())
                        .cwiseProduct(((1 - a_prev.array()).log()).matrix());
  double cost = (left_side + right_side).sum();
  return (-1) * cost / double(num_samples);
}

void BinaryCrossEntropyLoss::backpropagate(const Eigen::MatrixXf &a_prev,
                                           const Eigen::VectorXi &label) {
  Eigen::RowVectorXi label_row = label.transpose();

  assert(a_prev.cols() == label_row.size() && a_prev.rows() == 1 &&
         "Label and input dimensions do not match!");

  Eigen::RowVectorXf label_row_float = label_row.cast<float>();
  backprop_loss.array() = (-1) * label_row_float.array() / a_prev.array() +
                          (1 - label_row_float.array()) / (1 - a_prev.array());
}

const Eigen::MatrixXf &BinaryCrossEntropyLoss::get_backpropagate() const {
  return backprop_loss;
}
