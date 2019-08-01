#include "LossFunction/MultiCrossEntropyLoss.h"

double
MultiCrossEntropyLoss::calculate_loss(const Eigen::MatrixXf &a_prev,
                                      const Eigen::VectorXi &label) const {

  long label_size = label.size();

  assert(a_prev.cols() == label.size() &&
         "Number of labels does not match number of outputs");
  assert((std::abs(a_prev.col(0).sum() - 1.0) < 0.001) &&
         "Column sum not 1. Please use softmax activation for last layer");

  double error = 0.0;
  for (long i = 0; i < label_size; i++) {
    double temp = a_prev(label(i), i);
    error -= std::log(temp);
  }

  return error / (double)label_size;
}

void MultiCrossEntropyLoss::backpropagate(const Eigen::MatrixXf &a_prev,
                                          const Eigen::VectorXi &label) {

  // Check dimension
  const long number_samples = a_prev.cols();

  assert(number_samples == label.size() &&
         "Number of labels does not match number of outputs");
  assert((std::abs(a_prev.col(0).sum() - 1.0) < 0.001) &&
         "Column sum not 1. Please use softmax activation for last layer");

  backprop_loss.resize(a_prev.rows(), number_samples);
  backprop_loss.setZero();

  for (long i = 0; i < number_samples; i++) {
    backprop_loss(label(i), i) = float(-1.0) / a_prev(label(i), i);
  }
}

const Eigen::MatrixXf &MultiCrossEntropyLoss::get_backpropagate() const {
  return backprop_loss;
}
