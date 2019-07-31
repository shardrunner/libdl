#pragma once

#include "ActivationFunction/ActivationFunction.h"

/**
 * Represents an identity function as activation function
 * This function is primarily intended to be used for debugging, because it doesn't change the values.
 */
class IdentityFunction : public ActivationFunction {
public:
    void forward_propagation(Eigen::MatrixXf &input) const override;

    [[nodiscard]] Eigen::MatrixXf
    backward_propagation(const Eigen::MatrixXf &m_a,
                         const Eigen::MatrixXf &dC_da) const override;
};