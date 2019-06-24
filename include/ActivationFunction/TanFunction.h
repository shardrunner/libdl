#pragma once

#include "ActivationFunction.h"

class TanFunction : public ActivationFunction {
public:
  Eigen::MatrixXf apply_function(const Eigen::MatrixXf &input) const override;
  Eigen::MatrixXf apply_derivate(const Eigen::MatrixXf &input) const override;
};