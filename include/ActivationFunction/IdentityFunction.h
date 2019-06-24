#pragma once

#include "ActivationFunction/ActivationFunction.h"

class IdentityFunction : public ActivationFunction {
public:
  Eigen::MatrixXf apply_function(const Eigen::MatrixXf &input) const override;
  Eigen::MatrixXf apply_derivate(const Eigen::MatrixXf &input) const override;
};