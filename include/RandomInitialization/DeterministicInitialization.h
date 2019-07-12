#pragma once

#include "RandomInitialization.h"

/**
 * Generates random numbers deterministically by using C++ srand() with default
 * values and seed 0. Used for debugging purposes.
 */
class DeterministicInitialization : public RandomInitialization {
public:
  void initialize(Eigen::Ref<Eigen::MatrixXf> input) const override;
};