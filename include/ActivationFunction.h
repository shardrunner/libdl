#pragma once

#include <Eigen/Core>


/**
 * Represents an activation function
 */
class ActivationFunction {
public:
  virtual ~ActivationFunction() = default;
  /**
   * Applies the activation function to the input matrix
   * @param input The input matrix
   */
  virtual void apply_function(Eigen::MatrixXf &input) = 0;

  /**
   * Applies the derivate of the activation function to the input matrix
   * @param input The input matrix
   */
  virtual void apply_derivate(Eigen::MatrixXf &input) = 0;

};