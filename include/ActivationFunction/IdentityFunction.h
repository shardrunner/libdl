#pragma once

#include "ActivationFunction/ActivationFunction.h"

/**
 * Represents an identity function as activation function. Used for debugging.
 */
class IdentityFunction : public ActivationFunction {
public:
  void
  apply_function(Eigen::MatrixXf &input) const override;
  [[nodiscard]] Eigen::MatrixXf
  apply_derivative(const Eigen::MatrixXf &m_a,
                   const Eigen::MatrixXf &dC_da) const override;
};