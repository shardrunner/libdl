#pragma once

#include "RandomInitialization/RandomInitialization.h"

/**
 * Gaussian He initialization used best for Relu activation functions.
 */
class HetalInitialization : public RandomInitialization {
public:
  /**
   * [See abstract base class](@ref RandomInitialization)
   */
  void initialize(Eigen::Ref<Eigen::MatrixXf> input) const override;
};