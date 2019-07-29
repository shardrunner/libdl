#pragma once

#include "LossFunction/LossFunction.h"

/**
 * Calculates the cross entropy loss for multi class problems.
 * This is useful for multiclass classification.
 */
class MultiCrossEntropyLoss : public LossFunction {
public:
    /**
 * Calculates the loss for the given input x an label y using multiclass cross entropy loss.
 * @param a_prev Feed forward input from the previous layer.
 * @param label Corresponding labels as column Vector with 0..#classes per sample/row.
 * @return The calculated loss normalized with the number of samples
 */
  [[nodiscard]] double
  calculate_loss(const Eigen::MatrixXf &a_prev,
                 const Eigen::VectorXi &label) const override;

    /**
   * Calculates the derivative of the multiclass cross entropy loss with the given input x and labels y.
   * Stores it internally.
   * @param a_prev Feed forward input from the previous layer.
   * @param label Corresponding labels as column vector with 0..#classes per sample/row.
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