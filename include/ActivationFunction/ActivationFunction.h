#pragma once

#include <Eigen/Core>

/**
 * Abstract base class of an activation function.
 * An activation function is handed over to a layer class and used to activate the layer.
 * This is done in order to introduce non-linearities in the network.
 * Without the non-linearities the network would be easier to solve, but also far less powerful.
 * The class defines to virtual classes used for forwardpropagation and backpropagation.
 */
class ActivationFunction {
public:
    virtual ~ActivationFunction() = default;

/**
 * Applies the forward propagation step for the activation function.
 * The the represented activation function is applied coeffwise to the input.
 * This function is applied in the forward step of the layers.
 * After the evaluation with their respective weights, the values have to be activated in order to introduce non-linearities.
 * @param input The matrix at which the activation function is applied coeffwise.
 */
    virtual void
    forward_propagation(Eigen::MatrixXf &input) const = 0;

    /**
     * Applies backpropagation step for the activation function.
     * Calculates the derivative of the activation function in respect to the activated
     * input of the forward propagation and multiplies it with the derivative of the
     * next layer.
     * @param m_a The activated input matrix of the forward propagation of the previous layer. It was cached to evade a new evaluation.
     * @param dC_da The derivative of the next layer.in respect to the error.
     * @return dC_dz The derivative of the input of the activation function in respect to the error.
     */
    [[nodiscard]] virtual Eigen::MatrixXf
    backward_propagation(const Eigen::MatrixXf &m_a,
                         const Eigen::MatrixXf &dC_da) const = 0;
};