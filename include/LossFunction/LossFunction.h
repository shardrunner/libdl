#pragma once

#include <Eigen/Core>

/**
 * Loss function abstract class for the neural network. Used to minimize the
 * loss.
 */
class LossFunction {
public:
  /**
   * Virtual default destructor.
   */
  virtual ~LossFunction() = default;
  // TODO Change Row vectors for multi classification
  /**
   * Calculates the loss in the feed forward step.
   * @param a_prev The feed forward result of the previous layer
   * @param label The ground truth label set for the training input.
   * @return The calculated normalized loss for the set of training images.
   */
  [[nodiscard]] virtual float
  calculate_loss(const Eigen::MatrixXf &a_prev,
                 const Eigen::VectorXi &label) const = 0;

  /**
   * Calculates the derivative of the loss function in repsect to the input and
   * the label in the backpropagation step. The result is saved internally.
   * @param a_prev The feed forward result of the previous layer.
   * @param label The ground truth label set for the training input.
   */
  virtual void backpropagate(const Eigen::MatrixXf &a_prev,
                             const Eigen::VectorXi &label) = 0;

  /**
   * Get the internally saved result of the backpropagation step.
   * @return The derivative of the loss function calculated in the
   * backpropagate().
   */
  [[nodiscard]] virtual const Eigen::MatrixXf &get_backpropagate() const = 0;

protected:
  Eigen::MatrixXf temp_loss;
};