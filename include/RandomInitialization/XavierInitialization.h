#pragma once

#include "RandomInitialization/RandomInitialization.h"

/**
 * Gaussian Xavier initialization used for tanh, softmax and sigmoid activation functions.
 */
class XavierInitialization : public RandomInitialization {
public:
  void initialize(Eigen::Ref<Eigen::MatrixXf> input) const override;
};