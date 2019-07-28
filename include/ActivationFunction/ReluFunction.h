#pragma once

#include "ActivationFunction/ActivationFunction.h"

/**
 * Represents a Relu function as activation function. Can be a leaky relu.
 */
class ReluFunction : public ActivationFunction {
  /**
   * The leak value for the leaky relu. Defaults to 0.
   */
  float leak_factor;

public:
  explicit ReluFunction(float leak_factor = 0.0);
  void
  apply_function(Eigen::MatrixXf &input) const override;
  [[nodiscard]] Eigen::MatrixXf
  apply_derivative(const Eigen::MatrixXf &m_a,
                   const Eigen::MatrixXf &dC_da) const override;
};