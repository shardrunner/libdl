#pragma once

#include <Eigen/Core>

// TODO: Implement move constructor for derivate
/**
 * Represents an activation function
 */
class ActivationFunction {
public:
  virtual ~ActivationFunction() = default;
  /**
   * Applies the activation function on a copy to the input matrix in the
   * forward propagation.
   * @param input The input matrix
   * @return The activated input matrix
   */
  [[nodiscard]] virtual Eigen::MatrixXf
  apply_function(const Eigen::MatrixXf &input) const = 0;

  /**
   * Calculates the derivative of the activation function in respect to the
   * input of the forward propagation and multiply it with the derivative of the
   * next layer.
   * @param m_a The activated input matrix of the forward propagation. Used to
   * save computation time.
   * @param dC_da The derivative of the next layer.
   * @return dC_dz
   */
  [[nodiscard]] virtual Eigen::MatrixXf
  apply_derivative(const Eigen::MatrixXf &m_a,
                   const Eigen::MatrixXf &dC_da) const = 0;
};