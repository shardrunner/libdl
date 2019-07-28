#pragma once

#include "ActivationFunction/ActivationFunction.h"

/**
 * Represents a softmax activation function. Use before multiclass Cross Entropy
 * Loss.
 */
class SoftmaxFunction : public ActivationFunction {
public:
  void
  apply_function(Eigen::MatrixXf &input) const override;
  [[nodiscard]] Eigen::MatrixXf
  apply_derivative(const Eigen::MatrixXf &m_a,
                   const Eigen::MatrixXf &dC_da) const override;
};