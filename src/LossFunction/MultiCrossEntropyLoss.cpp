#include "LossFunction/MultiCrossEntropyLoss.h"

#include <iostream>

double MultiCrossEntropyLoss::calculate_loss(
        const Eigen::MatrixXf &a_prev, const Eigen::VectorXi &label) const {

    long label_size = label.size();

    assert(a_prev.cols() == label.size() && "Number of labels does not match number of outputs");

    double error = 0.0;
    for (long i = 0; i < label_size; i++) {
        double temp = a_prev(label(i), i);
        error -= std::log(temp);
    }

    return error / (double) label_size;
}

void MultiCrossEntropyLoss::backpropagate(const Eigen::MatrixXf &a_prev,
                                          const Eigen::VectorXi &label) {

    // Check dimension
    const long number_samples = a_prev.cols();

    assert(number_samples == label.size() && "Number of labels does not match number of outputs");

    /*if (label.size() != number_samples) {
      throw std::invalid_argument(
          "[class MultiClassEntropy]: Label and input do have incorrect dimensions");
    }*/

    backprop_loss.resize(a_prev.rows(), number_samples);
    backprop_loss.setZero();

    for (long i = 0; i < number_samples; i++) {
        backprop_loss(label(i), i) = float(-1.0) / a_prev(label(i), i);
    }
}

const Eigen::MatrixXf &MultiCrossEntropyLoss::get_backpropagate() const {
    return backprop_loss;
}
