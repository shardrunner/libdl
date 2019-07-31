#pragma once

#include "ActivationFunction/ActivationFunction.h"
//TODO relu kill vs depth
/**
 * Represents a Relu function as activation function.
 * This can also be a leaky Relu.
 * The Relu activation function is mainly used for convolutional layers.
 * They allow for deeper networks, without the gradients dying.
 * For the default of 0, values smaller than 0 are set to 0. This can lead to undesirable behaviour in the form of dead gradients. To circumvent this issue LeakyRelu has a value a bit smaller than 0 for values smaller than 0.
 *
 */
class ReluFunction : public ActivationFunction {
public:
    /**
     * The constructor for the (Leaky)Relu function.
     * This constructor allows the definition of a leak factor for the Relu, making it a LeakyRelu.
     * @param leak_factor The leak factor controls the Relu behaviour for values smaller than 0.
     */
    explicit ReluFunction(float leak_factor = 0.0);

    void forward_propagation(Eigen::MatrixXf &input) const override;

    [[nodiscard]] Eigen::MatrixXf
    backward_propagation(const Eigen::MatrixXf &m_a,
                         const Eigen::MatrixXf &dC_da) const override;

private:
    /**
     * The leak value for the leaky Relu. Defaults to 0.
     */
    float leak_factor;
};