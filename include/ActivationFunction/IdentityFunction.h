#pragma once

#include "ActivationFunction/ActivationFunction.h"

/**
 * Represents an identity function as activation function.
 *
 * This function is primarily intended to be used for debugging, because it
 * doesn't change the values.
 */
class IdentityFunction : public ActivationFunction {
public:
    /**
     * [See abstract base class](@ref ActivationFunction)
     *
     * do nothing...
     */
    void forward_propagation(Eigen::MatrixXf &input) const override;

    /**
   * [See abstract base class](@ref ActivationFunction)
     *
     * do nothing...
   */
    [[nodiscard]] Eigen::MatrixXf
    apply_derivative(const Eigen::MatrixXf &m_a,
                     const Eigen::MatrixXf &dC_da) const override;
};