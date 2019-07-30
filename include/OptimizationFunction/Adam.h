#pragma once

#include "OptimizationFunction/OptimizationFunction.h"
#include <Eigen/Core>

//http://ruder.io/optimizing-gradient-descent/
class Adam : public OptimizationFunction {
public:
    Adam(int filter_height, int filter_width, int bias_size);

    Adam(float learning_rate, float epsilon, float beta_1, float beta_2, int filter_height, int filter_width,
         int bias_size);

    void optimize_weights(Eigen::MatrixXf &values, const Eigen::MatrixXf &derivatives) override;

    void optimize_bias(Eigen::VectorXf &values, const Eigen::VectorXf &derivatives) override;

private:
    float m_learning_rate;
    float m_epsilon;
    float m_beta_1;
    float m_beta_2;

    Eigen::MatrixXf m_mt;
    Eigen::MatrixXf m_vt;
    Eigen::VectorXf m_mt_bias;
    Eigen::VectorXf m_vt_bias;
};