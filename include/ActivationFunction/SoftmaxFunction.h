#pragma once

#include "ActivationFunction/ActivationFunction.h"

/**
 * Represents a softmax activation function.
 *
 * This function normalizes all the input values.
 * After the softmax the column sum is 1.
 * This property makes it a good activation function before the
 * MulticlassCrossEntropy loss.
 */
class SoftmaxFunction : public ActivationFunction {
public:
    /**
* [See abstract base class](@ref ActivationFunction)
     *
     * A slight variation of simple softmax is used to reduce numerical instability.
*/
    void forward_propagation(Eigen::MatrixXf &input) const override;

    /**
 * [See abstract base class](@ref ActivationFunction)
 */
    [[nodiscard]] Eigen::MatrixXf
    apply_derivative(const Eigen::MatrixXf &m_a,
                     const Eigen::MatrixXf &dC_da) const override;
};