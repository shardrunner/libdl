#pragma once

#include "ActivationFunction/ActivationFunction.h"

/**
 * Represents a softmax activation function.
 * This function normalizes all the input values.
 * After the softmax the column sum is 1.
 * This property makes it a good activation function before the MulticlassCrossEntropy Loss.
 */
class SoftmaxFunction : public ActivationFunction {
public:
    void
    forward_propagation(Eigen::MatrixXf &input) const override;

    [[nodiscard]] Eigen::MatrixXf
    backward_propagation(const Eigen::MatrixXf &m_a,
                         const Eigen::MatrixXf &dC_da) const override;
};