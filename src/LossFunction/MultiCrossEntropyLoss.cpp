#include "LossFunction/MultiCrossEntropyLoss.h"
#include <cmath>

float MultiCrossEntropyLoss::calculate_loss(
    const Eigen::MatrixXf &a_prev, const Eigen::VectorXi &label) const {
  float error = 0.0;
  for (int i = 0; i < label.size(); i++) {
    error -= std::log(a_prev(label(i), i));
  }
}
void MultiCrossEntropyLoss::backpropagate(const Eigen::MatrixXf &a_prev,
                                          const Eigen::VectorXi &label) {}
const Eigen::MatrixXf &MultiCrossEntropyLoss::get_backpropagate() const {
  return temp_loss;
}
