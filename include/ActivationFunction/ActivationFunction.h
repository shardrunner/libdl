#pragma once

#include <Eigen/Core>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"


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
  virtual Eigen::MatrixXf apply_function(const Eigen::MatrixXf &input) const = 0;

  /**
   * Applies the derivate of the activation function to the input matrix
   * @param m_a The input matrix
   */
  virtual Eigen::MatrixXf
  apply_derivate(const Eigen::MatrixXf &m_a,
                 const Eigen::MatrixXf &dC_da) const = 0;

};