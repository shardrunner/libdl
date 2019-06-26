#pragma once

#include "RandomInitialization/RandomInitialization.h"

class HetalInitialization : public RandomInitialization {
public:
  void initialize(Eigen::Ref<Eigen::MatrixXf> input) const override;
};