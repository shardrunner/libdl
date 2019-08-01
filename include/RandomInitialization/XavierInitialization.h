#pragma once

#include "RandomInitialization/RandomInitialization.h"

/**
 * Gaussian Xavier initialization used for tanh, softmax and sigmoid activation
 * functions.
 */
class XavierInitialization : public RandomInitialization {
public:
  /**
   * [See abstract base class](@ref RandomInitialization)
   */
  void initialize(Eigen::Ref<Eigen::MatrixXf> input) const override;
};