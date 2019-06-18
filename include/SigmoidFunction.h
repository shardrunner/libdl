#pragma once

#include "ActivationFunction.h"

class SigmoidFunction : public ActivationFunction {
public:
  void apply_function(Eigen::MatrixXf &input) override;
  void apply_derivate(Eigen::MatrixXf &input) override;
};