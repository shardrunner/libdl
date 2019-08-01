#pragma once

#include <Eigen/Core>

/**
 * Abstract base class for optimization functions.
 */
class OptimizationFunction {
public:
    /**
     * Default virtual destructor.
     */
    virtual ~OptimizationFunction() = default;

    /**
     * Updates the given weights with the given derivative.
     *
     * The concrete implementation depends on the derived class.
     * @param values The weights to update.
     * @param derivatives The derivatives of the weights.
     */
    virtual void optimize_weights(Eigen::MatrixXf &values,
                                  const Eigen::MatrixXf &derivatives) = 0;

    /**
   * Updates the given bias with the given derivative.
   *
   * The concrete implementation depends on the derived class.
   * @param values The bias to update.
   * @param derivatives The derivatives of the bias.
   */
    virtual void optimize_bias(Eigen::VectorXf &values,
                               const Eigen::VectorXf &derivatives) = 0;
};