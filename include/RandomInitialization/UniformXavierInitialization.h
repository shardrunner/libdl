#pragma once

#include "RandomInitialization/RandomInitialization.h"

/**
 * Uniform Xavier initialization used for tanh activation functions.
 */
class UniformXavierInitialization : public RandomInitialization {
public:
  void initialize(Eigen::Ref<Eigen::MatrixXf> input) const override;
};
