#pragma once

#include "ActivationFunction/ActivationFunction.h"

/**
 * Represents a tanh activation function.
 */
class TanhFunction : public ActivationFunction {
public:
  [[nodiscard]] Eigen::MatrixXf
  apply_function(const Eigen::MatrixXf &input) const override;
  [[nodiscard]] Eigen::MatrixXf
  apply_derivative(const Eigen::MatrixXf &m_a,
                   const Eigen::MatrixXf &dC_da) const override;
};