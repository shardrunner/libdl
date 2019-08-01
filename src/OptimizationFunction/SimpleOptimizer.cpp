#include "OptimizationFunction/SimpleOptimizer.h"

void SimpleOptimizer::optimize_weights(Eigen::MatrixXf &values,
                                       const Eigen::MatrixXf &derivatives) {
  values -= m_learning_rate * derivatives;
}

void SimpleOptimizer::optimize_bias(Eigen::VectorXf &values,
                                    const Eigen::VectorXf &derivatives) {
  values -= m_learning_rate * derivatives;
}

SimpleOptimizer::SimpleOptimizer(float learning_rate)
    : m_learning_rate(learning_rate) {}
