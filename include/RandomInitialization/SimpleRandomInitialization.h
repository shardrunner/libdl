#pragma once

#include "RandomInitialization/RandomInitialization.h"

/**
 * Generates the random values using the C++ srand() default values and the
 * current time as seed.
 */
class SimpleRandomInitialization : public RandomInitialization {
public:
  void initialize(Eigen::Ref<Eigen::MatrixXf> input) const override;
};