#pragma once

#include "RandomInitialization/RandomInitialization.h"

/**
 * Gaussian He initialization used for tanh activation functions.
 */
class HetalInitialization : public RandomInitialization {
public:
  void initialize(Eigen::Ref<Eigen::MatrixXf> input) const override;
};