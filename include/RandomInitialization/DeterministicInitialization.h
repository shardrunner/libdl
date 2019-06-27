#pragma once

#include "RandomInitialization.h"
#include <Eigen/Core>

class DeterministicInitialization : public RandomInitialization {
public:
  void initialize(Eigen::Ref<Eigen::MatrixXf> input) const override;
};