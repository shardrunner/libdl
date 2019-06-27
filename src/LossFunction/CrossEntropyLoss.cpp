#include "LossFunction/CrossEntropyLoss.h"

float CrossEntropyLoss::calculate_loss(
    const Eigen::MatrixXf &a_prev,
    const Eigen::VectorXi &label) const { // const Eigen::MatrixXf &a_prev,
  //    const Eigen::RowVectorXf &label) const {
  return 1; //(-1)*(label.array()*Eigen::log(a_prev.array())).sum();
}
void CrossEntropyLoss::backpropagate(const Eigen::MatrixXf &a_prev,
                                     const Eigen::VectorXi &label) {}
const Eigen::MatrixXf &CrossEntropyLoss::get_backpropagate() const {
  return temp_loss;
}
