#pragma once

#include "RandomInitialization/RandomInitialization.h"

/**
 * Gaussian Xavier initialization used for tanh activation functions.
 */
class XavierInitialization : public RandomInitialization {
public:
  void initialize(Eigen::Ref<Eigen::MatrixXf> input) const override;
};