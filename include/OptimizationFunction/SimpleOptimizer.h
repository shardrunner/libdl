#pragma once

#include "OptimizationFunction/OptimizationFunction.h"



class SimpleOptimizer : public OptimizationFunction {
public:
    SimpleOptimizer(const float learning_rate);
    void optimize_weights(Eigen::MatrixXf &values, const Eigen::MatrixXf &derivatives) override;

    void optimize_bias(Eigen::VectorXf &values, const Eigen::VectorXf &derivatives) override;
private:
    const float m_learning_rate;
};