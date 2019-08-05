#pragma once

#include "LossFunction/LossFunction.h"

/**
 * Calculates the binary cross entropy loss for binary problems.
 * This is useful for binary classification.
 */
class BinaryCrossEntropyLoss : public LossFunction {
public:
  /**
   * Calculates the loss for the given input x and label y using binary cross
   * entropy loss.
   * -Sum(y+log(x)+(1-y)*log(1-x))/num_samples
   * @param a_prev Feed forward input from the previous layer.
   * @param label Corresponding labels as column Vector with 0/1 per sample/row.
   * @return The calculated loss normalized with the number of samples
   */
  [[nodiscard]] double
  calculate_loss(const Eigen::MatrixXf &a_prev,
                 const Eigen::VectorXi &label) const override;

  /**
   * Calculates the derivative of the binary cross entropy loss with the given
   * input x and labels y. Stores it internally. -y/x+(1-y)/(1-x)
   * @param a_prev Feed forward input from the previous layer.
   * @param label Corresponding labels as column vector with 0/1 per sample/row.
   */
  void backpropagate(const Eigen::MatrixXf &a_prev,
                     const Eigen::VectorXi &label) override;

  /**
   * Get the internally saved result of the backpropagation step.
   * @return The derivative of the loss function calculated in the
   * backpropagate().
   */
  [[nodiscard]] const Eigen::MatrixXf &get_backpropagate() const override;
};
