#pragma once

#include "ActivationFunction/ActivationFunction.h"

/**
 * Represents a sigmoid function as activation function.
 *
 * The sigmoid has the property to squash values between 0 and 1.
 */
class SigmoidFunction : public ActivationFunction {
public:
    /**
* [See abstract base class](@ref ActivationFunction)
*/
    void forward_propagation(Eigen::MatrixXf &input) const override;

    /**
 * [See abstract base class](@ref ActivationFunction)
 */
    [[nodiscard]] Eigen::MatrixXf
    apply_derivative(const Eigen::MatrixXf &m_a,
                     const Eigen::MatrixXf &dC_da) const override;
};