#pragma once

#include "RandomInitialization/RandomInitialization.h"

/**
 * Uniform He initialization used for Relu activation functions.
 */
class UniformHeInitialization : public RandomInitialization {
public:
  void initialize(Eigen::Ref<Eigen::MatrixXf> input) const override;
};
