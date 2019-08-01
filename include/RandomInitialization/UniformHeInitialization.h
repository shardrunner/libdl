#pragma once

#include "RandomInitialization/RandomInitialization.h"

/**
 * Uniform He initialization used for Relu activation functions.
 */
class UniformHeInitialization : public RandomInitialization {
public:
  /**
   * [See abstract base class](@ref RandomInitialization)
   */
  void initialize(Eigen::Ref<Eigen::MatrixXf> input) const override;
};
