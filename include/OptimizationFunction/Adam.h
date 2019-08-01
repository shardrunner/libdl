#pragma once

#include "OptimizationFunction/OptimizationFunction.h"

#include <Eigen/Core>

/**
 * Adam is a optimizer that combines first and second order momentum.
 *
 * To achieve this, it stores the values from the last update step and uses them
 * in combination with some hyperparameter.
 * // http://ruder.io/optimizing-gradient-descent/
 */
class Adam : public OptimizationFunction {
public:
  /**
   * Constructor
   * @param filter_height The height of the filter which the optimizer updates.
   * Used for storing the history.
   * @param filter_width The width of the filter which the optimizer updates.
   * Used for storing the history.
   * @param bias_size The bias size which the optimizer updates. Used for
   * storing the history.
   */
  Adam(int filter_height, int filter_width, int bias_size);

  /**
   * Constructor
   * @param filter_height The height of the filter which the optimizer updates.
   * Used for storing the history.
   * @param filter_width The width of the filter which the optimizer updates.
   * Used for storing the history.
   * @param bias_size The bias size which the optimizer updates. Used for
   * storing the history.
   * @param learning_rate The learning rate.
   * @param epsilon The epsilon should prevent division by zero.
   * @param beta_1 The hyperparameter for the first order momentum.
   * @param beta_2 The hyperparameter for the second order momentum.
   */
  Adam(float learning_rate, float epsilon, float beta_1, float beta_2,
       int filter_height, int filter_width, int bias_size);

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
  float m_learning_rate;
  float m_epsilon;
  float m_beta_1;
  float m_beta_2;

  Eigen::MatrixXf m_mt;
  Eigen::MatrixXf m_vt;
  Eigen::VectorXf m_mt_bias;
  Eigen::VectorXf m_vt_bias;
};