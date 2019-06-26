#pragma once

#include "RandomInitialization/RandomInitialization.h"

class XavierInitialization : public RandomInitialization {
public:
  void initialize(Eigen::Ref<Eigen::MatrixXf> input) const override;
};