#pragma once

#include "ActivationFunction/ActivationFunction.h"

/**
 * Represents an identity function as activation function. Used for debugging.
 */
class IdentityFunction : public ActivationFunction {
public:
  [[nodiscard]] Eigen::MatrixXf
  apply_function(const Eigen::MatrixXf &input) const override;
  [[nodiscard]] Eigen::MatrixXf
  apply_derivative(const Eigen::MatrixXf &m_a,
                   const Eigen::MatrixXf &dC_da) const override;
};