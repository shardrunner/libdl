#pragma once

#include "OptimizationFunction/OptimizationFunction.h"

/**
 * A simple Stochastic Gradient Descend optimizer.
 *
 * It uses a fixed learning rate to optimize.
 */
class SimpleOptimizer : public OptimizationFunction {
public:
  /**
   * Constructor.
   * @param learning_rate The fixed learning rate of the optimizer.
   */
  SimpleOptimizer(const float learning_rate);

  /**
   * [See abstract base class](@ref OptimizationFunction)
   */
  void optimize_weights(Eigen::MatrixXf &values,
                        const Eigen::MatrixXf &derivatives) override;

  /**
   * [See abstract base class](@ref OptimizationFunction)
   */
  void optimize_bias(Eigen::VectorXf &values,
                     const Eigen::VectorXf &derivatives) override;

private:
  const float m_learning_rate;
};