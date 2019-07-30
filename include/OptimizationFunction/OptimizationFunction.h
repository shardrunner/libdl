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

    virtual void optimize_weights(Eigen::MatrixXf &values, const Eigen::MatrixXf &derivatives) =0;

    virtual void optimize_bias(Eigen::VectorXf &values, const Eigen::VectorXf &derivatives) =0;
};