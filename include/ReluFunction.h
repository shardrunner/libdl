#pragma once

#include "ActivationFunction.h"

class ReluFunction : public ActivationFunction {
  float leak_factor;
public:
  ReluFunction(float leak_factor=0);
  void apply_function(Eigen::MatrixXf &input) override;
  void apply_derivate(Eigen::MatrixXf &input) override;
};