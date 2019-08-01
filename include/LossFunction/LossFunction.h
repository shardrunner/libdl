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

    /**
     * Calculates the loss in the feed forward step.
     * @param a_prev The feed forward result of the previous layer
     * @param label The ground truth label set for the training input.
     * @return The calculated normalized loss for the set of training images.
     */
    [[nodiscard]] virtual double
    calculate_loss(const Eigen::MatrixXf &a_prev,
                   const Eigen::VectorXi &label) const = 0;

    /**
     * Calculates the derivative of the loss function in respect to the input and
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
    Eigen::MatrixXf backprop_loss;
};