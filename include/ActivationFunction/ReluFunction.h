#pragma once

#include "ActivationFunction.h"

class ReluFunction : public ActivationFunction {
  float leak_factor;
public:
  explicit ReluFunction(float leak_factor=0.0);
  Eigen::MatrixXf apply_function(const Eigen::MatrixXf &input) const override;
  Eigen::MatrixXf apply_derivate(const Eigen::MatrixXf &input) const override;
};